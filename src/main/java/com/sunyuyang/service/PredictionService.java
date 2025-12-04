package com.sunyuyang.service;

import com.sunyuyang.dao.StockDataDao;
import com.sunyuyang.dao.ZhituStockDataDao;
import com.sunyuyang.entity.ZhituStockKLine;
import com.sunyuyang.model.LSTMModel;
import com.sunyuyang.util.PredictionDenormalizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class PredictionService {
    private static final Logger logger = LoggerFactory.getLogger(PredictionService.class);
    private final ZhituStockDataDao stockDataDao;
    private final DataPreprocessingService preprocessingService;

    public PredictionService(ZhituStockDataDao stockDataDao, DataPreprocessingService preprocessingService) {
        this.stockDataDao = stockDataDao;
        this.preprocessingService = preprocessingService;
    }

    // 示例转换方法
    public LocalDateTime convertToLocalDateTime(String dateTimeStr) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"); // 根据实际格式调整
        return LocalDateTime.parse(dateTimeStr, formatter);
    }

    /**
     * 预测未来价格
     */
    public PredictionResult predictFuturePrices(String stockCode, LSTMModel model,
                                                int timeSteps, int predictSteps) {
        logger.info("Predicting future prices for stock: {}", stockCode);

        try {
            // 1. 获取最近的数据
            LocalDateTime endDate = LocalDateTime.now();
            LocalDateTime startDate = endDate.minusDays(timeSteps * 15 / (24 * 60)); // 近似计算

            List<ZhituStockKLine> recentData = stockDataDao.getKLineData(stockCode, startDate, endDate);

            if (recentData.size() < timeSteps) {
                throw new IllegalArgumentException(String.format(
                        "Not enough data. Required: %d, Available: %d",
                        timeSteps, recentData.size()));
            }

            // 2. 预处理数据
            DataPreprocessingService.ProcessedData processedData =
                    preprocessingService.preprocessData(recentData, timeSteps, predictSteps);

            // 3. 使用最新时间窗口进行预测
            INDArray latestWindow = processedData.getFeatures()
                    .getRow(processedData.getFeatures().size(0) - 1);

            // 重塑为 [1, timeSteps, features] 格式
            INDArray input = latestWindow.reshape(1, timeSteps, (int) latestWindow.size(1));

            // 4. 进行预测
            INDArray normalizedPredictions = model.getModel().output(input);

            // 5. 反标准化预测结果
            //INDArray predictions = processedData.getLabelNormalizer()
            //        .revert(normalizedPredictions);
            // 使用示例
            PredictionDenormalizer denormalizer = new PredictionDenormalizer(normalizedPredictions);
            INDArray predictions = denormalizer.denormalize(normalizedPredictions);

            // 6. 转换为实际价格
            double lastPrice = recentData.get(recentData.size() - 1).getClose();
            List<Double> futurePrices = convertToActualPrices(predictions, lastPrice);

            // 7. 生成预测时间点
            List<LocalDateTime> predictionTimes = generatePredictionTimes(
                    convertToLocalDateTime(recentData.get(recentData.size() - 1).getTradeTime()), predictSteps);

            logger.info("Prediction completed. Generated {} future price points", futurePrices.size());

            return new PredictionResult(futurePrices, predictionTimes, lastPrice);

        } catch (Exception e) {
            logger.error("Prediction failed for stock: {}", stockCode, e);
            throw new RuntimeException("Prediction failed", e);
        }
    }

    /**
     * 转换为实际价格
     */
    private List<Double> convertToActualPrices(INDArray normalizedPredictions, double lastPrice) {
        List<Double> prices = new ArrayList<>();

        // 假设预测的是价格变化率
        INDArray predictions = normalizedPredictions.getRow(0);

        double currentPrice = lastPrice;
        for (int i = 0; i < predictions.size(1); i++) {
            double changeRate = predictions.getDouble(0, i);
            double predictedPrice = currentPrice * (1 + changeRate);
            prices.add(predictedPrice);
            currentPrice = predictedPrice; // 累积预测
        }

        return prices;
    }

    /**
     * 生成预测时间点
     */
    private List<LocalDateTime> generatePredictionTimes(LocalDateTime lastTime, int steps) {
        List<LocalDateTime> times = new ArrayList<>();
        LocalDateTime current = lastTime;

        // 15分钟间隔
        for (int i = 1; i <= steps; i++) {
            current = current.plusMinutes(15);
            // 跳过非交易时间（简化处理）
            if (isTradingTime(current)) {
                times.add(current);
            }
        }

        return times;
    }

    /**
     * 检查是否为交易时间（简化版）
     */
    private boolean isTradingTime(LocalDateTime time) {
        int hour = time.getHour();
        int minute = time.getMinute();

        // 假设交易时间：9:30-11:30, 13:00-15:00
        boolean isMorning = (hour == 9 && minute >= 30) || (hour == 10) || (hour == 11 && minute <= 30);
        boolean isAfternoon = (hour == 13) || (hour == 14) || (hour == 15 && minute == 0);

        return isMorning || isAfternoon;
    }

    /**
     * 批量预测多个股票
     */
    public List<StockPrediction> batchPredict(List<String> stockCodes, LSTMModel model,
                                              int timeSteps, int predictSteps) {
        List<StockPrediction> predictions = new ArrayList<>();

        for (String stockCode : stockCodes) {
            try {
                PredictionResult result = predictFuturePrices(stockCode, model, timeSteps, predictSteps);
                predictions.add(new StockPrediction(stockCode, result));

                logger.info("Prediction for {} completed successfully", stockCode);

            } catch (Exception e) {
                logger.error("Failed to predict for stock: {}", stockCode, e);
                predictions.add(new StockPrediction(stockCode, e.getMessage()));
            }
        }

        return predictions;
    }

    /**
     * 预测结果
     */
    public static class PredictionResult {
        private final List<Double> futurePrices;
        private final List<LocalDateTime> predictionTimes;
        private final double lastActualPrice;
        private final LocalDateTime predictionTime;

        public PredictionResult(List<Double> futurePrices, List<LocalDateTime> predictionTimes,
                                double lastActualPrice) {
            this.futurePrices = futurePrices;
            this.predictionTimes = predictionTimes;
            this.lastActualPrice = lastActualPrice;
            this.predictionTime = LocalDateTime.now();
        }

        public void print() {
            System.out.println("\n=== 股价预测结果 ===");
            System.out.printf("预测时间: %s%n", predictionTime);
            System.out.printf("最后实际价格: %.4f%n", lastActualPrice);
            System.out.println("未来价格预测:");

            for (int i = 0; i < futurePrices.size(); i++) {
                double price = futurePrices.get(i);
                double change = ((price - lastActualPrice) / lastActualPrice) * 100;
                System.out.printf("  %s: %.4f (%.2f%%)%n",
                        predictionTimes.get(i), price, change);
            }

            double avgPrediction = futurePrices.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElse(lastActualPrice);

            System.out.printf("\n平均预测价格: %.4f%n", avgPrediction);
            System.out.printf("预期变化: %.2f%%%n",
                    ((avgPrediction - lastActualPrice) / lastActualPrice) * 100);
        }

        public List<Double> getFuturePrices() {
            return futurePrices;
        }

        public List<LocalDateTime> getPredictionTimes() {
            return predictionTimes;
        }

        public double getLastActualPrice() {
            return lastActualPrice;
        }
    }

    /**
     * 股票预测
     */
    public static class StockPrediction {
        private final String stockCode;
        private final PredictionResult predictionResult;
        private final String errorMessage;
        private final boolean success;

        public StockPrediction(String stockCode, PredictionResult predictionResult) {
            this.stockCode = stockCode;
            this.predictionResult = predictionResult;
            this.errorMessage = null;
            this.success = true;
        }

        public StockPrediction(String stockCode, String errorMessage) {
            this.stockCode = stockCode;
            this.predictionResult = null;
            this.errorMessage = errorMessage;
            this.success = false;
        }

        public boolean isSuccess() {
            return success;
        }

        public String getStockCode() {
            return stockCode;
        }

        public PredictionResult getPredictionResult() {
            return predictionResult;
        }

        public String getErrorMessage() {
            return errorMessage;
        }
    }
}
