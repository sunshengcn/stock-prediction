package com.sunyuyang.service;

import com.sunyuyang.entity.ModelConfig;
import com.sunyuyang.model.LSTMModel;
import com.sunyuyang.util.EvaluationMetrics;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ModelTrainingService {
    private static final Logger logger = LoggerFactory.getLogger(ModelTrainingService.class);
    private final ModelConfig config;

    public ModelTrainingService(ModelConfig config) {
        this.config = config;
    }

    /**
     * 训练模型 - 修正版本
     */
    public TrainingResult trainModel(INDArray features, INDArray labels, String modelName) {
        logger.info("Starting model training...");
        logger.info("Features shape: {}, Labels shape: {}", features.shape(), labels.shape());

        try {
            // 1. 检查数据有效性
            if (features.size(0) == 0 || labels.size(0) == 0) {
                throw new IllegalArgumentException("训练数据为空");
            }

            // 确保特征和标签样本数匹配
            int minSamples = Math.min((int) features.size(0), (int) labels.size(0));
            features = features.get(
                    NDArrayIndex.interval(0, minSamples),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
            labels = labels.get(
                    NDArrayIndex.interval(0, minSamples),
                    NDArrayIndex.all()
            );

            logger.info("对齐后数据: 特征shape={}, 标签shape={}", features.shape(), labels.shape());

            // 2. 划分训练集和测试集
            SplitResult splitResult = splitTrainTestSafe(features, labels, config.getTrainTestSplit());

            // 3. 初始化模型
            int timeSteps = (int) features.size(1); // 时间步长
            int numInputFeatures = (int) features.size(2); // 特征数
            int numOutputSteps = config.getPredictSteps(); // 输出步长

            logger.info("模型参数 - 时间步长: {}, 输入特征数: {}, 输出步长: {}",
                    timeSteps, numInputFeatures, numOutputSteps);

            LSTMModel lstmModel = new LSTMModel(config);
            lstmModel.initialize(numInputFeatures, numOutputSteps, timeSteps);
            MultiLayerNetwork model = lstmModel.getModel();

            // 4. 创建数据迭代器
            DataSetIterator trainIterator = createDataSetIterator(
                    splitResult.trainFeatures, splitResult.trainLabels, config.getBatchSize(), true);
            DataSetIterator testIterator = createDataSetIterator(
                    splitResult.testFeatures, splitResult.testLabels, config.getBatchSize(), false);

            // 5. 训练模型
            logger.info("开始模型训练...");
            int actualEpochs = Math.min(config.getEpochs(), 50);

            // 记录训练过程中的损失
            List<Double> epochLosses = new ArrayList<>();

            for (int epoch = 0; epoch < actualEpochs; epoch++) {
                // 训练一个epoch
                model.fit(trainIterator);
                trainIterator.reset();

                // 计算当前epoch的平均损失
                double epochLoss = calculateLoss(model, trainIterator);
                epochLosses.add(epochLoss);

                if ((epoch + 1) % 5 == 0) {
                    logger.info("训练轮次 {}/{} - 平均损失: {:.6f}",
                            epoch + 1, actualEpochs, epochLoss);
                }
            }

            // 6. 评估模型
            EvaluationResult evalResult = evaluateModel(model, splitResult);

            // 7. 保存最终模型
            lstmModel.saveModel(modelName);

            logger.info("模型训练完成，最终损失: {:.6f}",
                    epochLosses.isEmpty() ? 0.0 : epochLosses.get(epochLosses.size() - 1));

            return new TrainingResult(model, evalResult, null);

        } catch (Exception e) {
            logger.error("模型训练失败", e);
            throw new RuntimeException("模型训练失败", e);
        }
    }

    /**
     * 计算模型在数据集上的平均损失
     */
    private double calculateLoss(MultiLayerNetwork model, DataSetIterator iterator) {
        double totalLoss = 0.0;
        int batchCount = 0;

        iterator.reset();
        while (iterator.hasNext()) {
            DataSet batch = iterator.next();
            totalLoss += model.score(batch);  // score方法需要DataSet，不是DataSetIterator
            batchCount++;
        }

        iterator.reset();
        return batchCount > 0 ? totalLoss / batchCount : 0.0;
    }

    /**
     * 创建数据集迭代器 - 修正版本
     */
    private DataSetIterator createDataSetIterator(INDArray features, INDArray labels, int batchSize, boolean shuffle) {
        List<DataSet> dataSets = new ArrayList<>();

        if (features.size(0) == 0 || labels.size(0) == 0) {
            logger.warn("创建迭代器时发现空数据集");
            return new ListDataSetIterator<>(dataSets, batchSize);
        }

        int numSamples = (int) features.size(0);

        for (int i = 0; i < numSamples; i++) {
            // 获取特征 - 保持3D形状 [1, time_steps, features]
            INDArray feature = features.get(
                    NDArrayIndex.point(i),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            ).reshape(1, (int) features.size(1), (int) features.size(2));

            // 获取标签 - 2D形状 [1, output_steps]
            INDArray label = labels.get(
                    NDArrayIndex.point(i),
                    NDArrayIndex.all()
            ).reshape(1, (int) labels.size(1));

            dataSets.add(new DataSet(feature, label));
        }

        // 如果需要打乱数据
        if (shuffle) {
            java.util.Collections.shuffle(dataSets);
        }

        return new ListDataSetIterator<>(dataSets, Math.min(batchSize, dataSets.size()));
    }

    /**
     * 安全的划分训练集和测试集
     */
    private SplitResult splitTrainTestSafe(INDArray features, INDArray labels, double splitRatio) {
        int totalSamples = (int) features.size(0);
        int trainSize = (int) (totalSamples * splitRatio);

        // 确保trainSize不超过数组范围
        if (trainSize >= totalSamples) {
            trainSize = totalSamples - 1;
            logger.warn("调整trainSize: {} -> {}", (int) (totalSamples * splitRatio), trainSize);
        }

        // 确保至少有一个测试样本
        if (trainSize >= totalSamples - 1) {
            trainSize = totalSamples - 2;
        }

        logger.info("数据划分 - 总样本: {}, 训练集: {}, 测试集: {}",
                totalSamples, trainSize, totalSamples - trainSize);

        // 获取数据
        INDArray trainFeatures = getSafeRows(features, 0, trainSize);
        INDArray testFeatures = getSafeRows(features, trainSize, totalSamples);

        INDArray trainLabels = getSafeRows(labels, 0, trainSize);
        INDArray testLabels = getSafeRows(labels, trainSize, totalSamples);

        return new SplitResult(trainFeatures, trainLabels, testFeatures, testLabels);
    }

    /**
     * 安全获取行数据
     */
    private INDArray getSafeRows(INDArray array, int start, int end) {
        int arrayRows = (int) array.size(0);

        // 调整索引确保在范围内
        if (start < 0) start = 0;
        if (end > arrayRows) end = arrayRows;
        if (start >= end) {
            // 返回空数组
            if (array.rank() == 3) {
                return Nd4j.create(0, array.size(1), array.size(2));
            } else if (array.rank() == 2) {
                return Nd4j.create(0, array.size(1));
            } else {
                return Nd4j.create(0);
            }
        }

        // 根据数组维度选择正确的索引方式
        if (array.rank() == 3) {
            return array.get(
                    NDArrayIndex.interval(start, end),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
        } else if (array.rank() == 2) {
            return array.get(
                    NDArrayIndex.interval(start, end),
                    NDArrayIndex.all()
            );
        } else {
            return array.get(NDArrayIndex.interval(start, end));
        }
    }

    /**
     * 评估模型
     */
    private EvaluationResult evaluateModel(MultiLayerNetwork model, SplitResult splitResult) {
        try {
            // 训练集评估
            INDArray trainPredictions = model.output(splitResult.trainFeatures);
            INDArray trainActual = splitResult.trainLabels;

            // 测试集评估
            INDArray testPredictions = model.output(splitResult.testFeatures);
            INDArray testActual = splitResult.testLabels;

            return new EvaluationResult(trainPredictions, trainActual, testPredictions, testActual);
        } catch (Exception e) {
            logger.error("模型评估失败", e);
            // 返回空评估结果
            return new EvaluationResult(Nd4j.create(0), Nd4j.create(0),
                    Nd4j.create(0), Nd4j.create(0));
        }
    }

    /**
     * 内部类：划分结果
     */
    private static class SplitResult {
        final INDArray trainFeatures;
        final INDArray trainLabels;
        final INDArray testFeatures;
        final INDArray testLabels;

        SplitResult(INDArray trainFeatures, INDArray trainLabels,
                    INDArray testFeatures, INDArray testLabels) {
            this.trainFeatures = trainFeatures;
            this.trainLabels = trainLabels;
            this.testFeatures = testFeatures;
            this.testLabels = testLabels;
        }
    }

    /**
     * 训练结果
     */
    public static class TrainingResult {
        private final MultiLayerNetwork model;
        private final EvaluationResult evaluation;
        private final EarlyStoppingResult<MultiLayerNetwork> earlyStoppingResult;

        public TrainingResult(MultiLayerNetwork model, EvaluationResult evaluation,
                              EarlyStoppingResult<MultiLayerNetwork> earlyStoppingResult) {
            this.model = model;
            this.evaluation = evaluation;
            this.earlyStoppingResult = earlyStoppingResult;
        }

        public MultiLayerNetwork getModel() {
            return model;
        }

        public EvaluationResult getEvaluation() {
            return evaluation;
        }

        public EarlyStoppingResult<MultiLayerNetwork> getEarlyStoppingResult() {
            return earlyStoppingResult;
        }
    }

    /**
     * 评估结果
     */
    public static class EvaluationResult {
        final INDArray trainPredictions;
        final INDArray trainActual;
        final INDArray testPredictions;
        final INDArray testActual;

        public EvaluationResult(INDArray trainPredictions, INDArray trainActual,
                                INDArray testPredictions, INDArray testActual) {
            this.trainPredictions = trainPredictions;
            this.trainActual = trainActual;
            this.testPredictions = testPredictions;
            this.testActual = testActual;
        }

        public void printMetrics() {
            if (trainPredictions.size(0) == 0) {
                System.out.println("评估结果为空");
                return;
            }

            System.out.println("\n=== 训练集评估 ===");
            EvaluationMetrics.printAllMetrics(trainPredictions, trainActual, "训练集");

            System.out.println("\n=== 测试集评估 ===");
            EvaluationMetrics.printAllMetrics(testPredictions, testActual, "测试集");
        }
    }

    /**
     * 交叉验证结果
     */
    public static class CrossValidationResult {
        final double averageScore;
        final double stdScore;
        final List<Double> foldScores;

        public CrossValidationResult(double averageScore, double stdScore, List<Double> foldScores) {
            this.averageScore = averageScore;
            this.stdScore = stdScore;
            this.foldScores = foldScores;
        }

        public void print() {
            System.out.println("\n=== 交叉验证结果 ===");
            System.out.printf("平均损失: %.6f%n", averageScore);
            System.out.printf("标准差: %.6f%n", stdScore);
            System.out.println("各折得分: " + foldScores);
        }
    }
}