package com.sunyuyang;

import com.sunyuyang.dao.DatabaseConfig;
import com.sunyuyang.dao.ZhituStockDataDao;
import com.sunyuyang.entity.ModelConfig;
import com.sunyuyang.entity.ZhituStockKLine;
import com.sunyuyang.model.LSTMModel;
import com.sunyuyang.service.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;

public class StockPredictionApp {
    private static final Logger logger = LoggerFactory.getLogger(StockPredictionApp.class);

    // 配置
    private static final String STOCK_CODE = "300624.SZ"; // 示例股票代码
    private static final String MODEL_NAME = "stock_predictor_v1";
    private static final boolean RETRAIN_MODEL = true; // 是否重新训练模型

    public static void main(String[] args) {
        logger.info("Starting Stock Prediction Application...");

        try {
            // 1. 初始化组件
            ModelConfig modelConfig = ModelConfig.getDefaultConfig();
            ZhituStockDataDao stockDataDao = new ZhituStockDataDao();
            DataPreprocessingService preprocessingService = new DataPreprocessingService();
            ModelTrainingService trainingService = new ModelTrainingService(modelConfig);
            PredictionService predictionService = new PredictionService(stockDataDao, preprocessingService);

            // 2. 检查数据库连接
            if (!stockDataDao.checkTableExists()) {
                logger.error("Database table 'zhitu_stock_k_line' does not exist");
                System.err.println("请先创建数据库表（参考提供的SQL脚本）");
                return;
            }

            // 3. 获取数据
            LocalDateTime endDate = LocalDateTime.now();
            LocalDateTime startDate = endDate.minusYears(3); // 过去三年

            logger.info("Fetching data from {} to {}", startDate, endDate);
            List<ZhituStockKLine> klineData = stockDataDao.getKLineData(STOCK_CODE, startDate, endDate);

            if (klineData.isEmpty()) {
                logger.error("No data found for stock: {}", STOCK_CODE);
                System.err.println("没有找到股票数据，请检查数据库");
                return;
            }

            logger.info("Loaded {} records for training", klineData.size());

            // 4. 模型训练或加载
            LSTMModel lstmModel;

            if (RETRAIN_MODEL) {
                logger.info("Starting model training...");

                // 4.1 数据预处理
                DataPreprocessingService.ProcessedData processedData =
                        preprocessingService.preprocessData(klineData,
                                modelConfig.getTimeSteps(), modelConfig.getPredictSteps());

                // 4.2 训练模型
                ModelTrainingService.TrainingResult trainingResult =
                        trainingService.trainModel(
                                processedData.getFeatures(),
                                processedData.getLabels(),
                                MODEL_NAME);

                // 4.3 保存标准化器
                preprocessingService.saveNormalizers(
                        processedData.getFeatureNormalizer(),
                        processedData.getLabelNormalizer(),
                        MODEL_NAME);

                // 4.4 评估结果
                trainingResult.getEvaluation().printMetrics();

                lstmModel = new LSTMModel(modelConfig);
                lstmModel.initialize(
                        (int) processedData.getFeatures().size(2),
                        modelConfig.getPredictSteps());

                logger.info("Model training completed successfully");

            } else {
                logger.info("Loading existing model...");
                lstmModel = LSTMModel.loadModel(MODEL_NAME);

                // 加载标准化器
                DataPreprocessingService.NormalizerPair normalizers =
                        preprocessingService.loadNormalizers(MODEL_NAME);

                if (normalizers == null) {
                    logger.warn("Normalizers not found, predictions may be inaccurate");
                }

                logger.info("Model loaded successfully");
            }

            // 5. 打印模型信息
            System.out.println(lstmModel.getModelSummary());

            // 6. 进行预测
            logger.info("Making predictions for the next 5 trading days...");
            PredictionService.PredictionResult predictionResult =
                    predictionService.predictFuturePrices(
                            STOCK_CODE,
                            lstmModel,
                            modelConfig.getTimeSteps(),
                            modelConfig.getPredictSteps());

            // 7. 显示预测结果
            predictionResult.print();

            // 8. 可选：批量预测多个股票
            if (args.length > 0) {
                List<String> stockCodes = Arrays.asList(args);
                logger.info("Batch predicting for {} stocks", stockCodes.size());

                List<PredictionService.StockPrediction> batchResults =
                        predictionService.batchPredict(
                                stockCodes,
                                lstmModel,
                                modelConfig.getTimeSteps(),
                                modelConfig.getPredictSteps());

                // 打印批量结果
                System.out.println("\n=== 批量预测结果 ===");
                for (PredictionService.StockPrediction stockPrediction : batchResults) {
                    if (stockPrediction.isSuccess()) {
                        System.out.printf("%s: 预测成功，生成%d个预测点%n",
                                stockPrediction.getStockCode(),
                                stockPrediction.getPredictionResult().getFuturePrices().size());
                    } else {
                        System.out.printf("%s: 预测失败 - %s%n",
                                stockPrediction.getStockCode(),
                                stockPrediction.getErrorMessage());
                    }
                }
            }

            logger.info("Stock Prediction Application completed successfully");

        } catch (Exception e) {
            logger.error("Application failed", e);
            System.err.println("应用程序执行失败: " + e.getMessage());
        } finally {
            // 清理资源
            DatabaseConfig.closeDataSource();
        }
    }

    /**
     * 创建示例数据（用于测试）
     */
    private static void createSampleData(ZhituStockDataDao dao) {
        logger.info("Creating sample data for testing...");

        // 这里可以添加代码来生成或导入示例数据
        // 实际应用中应从数据源获取真实数据

        logger.info("Sample data creation complete");
    }
}
