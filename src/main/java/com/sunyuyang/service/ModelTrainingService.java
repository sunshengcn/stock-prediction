package com.sunyuyang.service;

import com.sunyuyang.entity.ModelConfig;
import com.sunyuyang.model.LSTMModel;
import com.sunyuyang.util.EvaluationMetrics;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class ModelTrainingService {
    private static final Logger logger = LoggerFactory.getLogger(ModelTrainingService.class);
    private final ModelConfig config;

    public ModelTrainingService(ModelConfig config) {
        this.config = config;
    }

    /**
     * 训练模型
     */
    public TrainingResult trainModel(INDArray features, INDArray labels, String modelName) {
        logger.info("Starting model training...");

        try {
            // 1. 划分训练集和测试集
            SplitResult splitResult = splitTrainTest(features, labels, config.getTrainTestSplit());

            // 2. 创建数据迭代器
            DataSetIterator trainIterator = createDataSetIterator(
                    splitResult.trainFeatures, splitResult.trainLabels, config.getBatchSize());

            DataSetIterator testIterator = createDataSetIterator(
                    splitResult.testFeatures, splitResult.testLabels, config.getBatchSize());

            // 3. 初始化模型
            int numInputFeatures = (int) features.size(2); // [样本数, 时间步长, 特征数]
            LSTMModel lstmModel = new LSTMModel(config);
            lstmModel.initialize(numInputFeatures, config.getPredictSteps());

            MultiLayerNetwork model = lstmModel.getModel();

            // 4. 配置早停

            // 使用其他可用的终止条件
            MaxEpochsTerminationCondition maxEpochs = new MaxEpochsTerminationCondition(100);
            MaxScoreIterationTerminationCondition maxScore = new MaxScoreIterationTerminationCondition(0.5);

            EarlyStoppingConfiguration<MultiLayerNetwork> esConfig =
                    new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                            .epochTerminationConditions(new MaxEpochsTerminationCondition(config.getEpochs()))
                            .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(Duration.ofHours(2).toMillis(), TimeUnit.MILLISECONDS))
                            .scoreCalculator(new DataSetLossCalculator(testIterator, true))
                            .evaluateEveryNEpochs(1)
                            .modelSaver(new LocalFileModelSaver("models/" + modelName))
                            .build();

            // 5. 使用早停训练
            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(
                    esConfig, model, trainIterator);

            EarlyStoppingResult<MultiLayerNetwork> esResult = trainer.fit();

            // 6. 加载最佳模型
            MultiLayerNetwork bestModel = esResult.getBestModel();

            // 7. 评估模型
            EvaluationResult evalResult = evaluateModel(bestModel, splitResult);

            // 8. 保存最终模型
            lstmModel.saveModel(modelName);

            logger.info("Model training completed successfully");

            return new TrainingResult(bestModel, evalResult, esResult);

        } catch (Exception e) {
            logger.error("Model training failed", e);
            throw new RuntimeException("Model training failed", e);
        }
    }

    /**
     * 创建数据集迭代器
     */
    private DataSetIterator createDataSetIterator(INDArray features, INDArray labels, int batchSize) {
        List<DataSet> dataSets = new ArrayList<>();

        for (int i = 0; i < features.size(0); i++) {
            INDArray feature = features.getRow(i);
            INDArray label = labels.getRow(i);
            dataSets.add(new DataSet(feature, label));
        }

        return new ListDataSetIterator<>(dataSets, batchSize);
    }

    /**
     * 划分训练集和测试集
     */
    /*
    private SplitResult splitTrainTest(INDArray features, INDArray labels, double splitRatio) {
        int totalSamples = (int) features.size(0);
        int trainSize = (int) (totalSamples * splitRatio);

        // 时间序列数据，按时间顺序划分
        INDArray trainFeatures = features.get(Nd4j.createIndexArray(0, trainSize), Nd4j.all(), Nd4j.all());
        INDArray trainLabels = labels.get(Nd4j.createIndexArray(0, trainSize), Nd4j.all());

        INDArray testFeatures = features.get(Nd4j.createIndexArray(trainSize, totalSamples),
                Nd4j.all(), Nd4j.all());
        INDArray testLabels = labels.get(Nd4j.createIndexArray(trainSize, totalSamples), Nd4j.all());

        logger.info("Data split - Train: {} samples, Test: {} samples",
                trainSize, totalSamples - trainSize);

        return new SplitResult(trainFeatures, trainLabels, testFeatures, testLabels);
    }*/
    private SplitResult splitTrainTest(INDArray features, INDArray labels, double splitRatio) {
        int totalSamples = (int) features.size(0);
        int trainSize = (int) (totalSamples * splitRatio);

        // 根据 features 和 labels 的维度选择正确的索引方式

        if (features.rank() == 3 && labels.rank() == 3) {
            // 如果都是3D数组（样本数, 时间步长, 特征数）
            INDArray trainFeatures = features.get(
                    NDArrayIndex.interval(0, trainSize),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
            INDArray trainLabels = labels.get(
                    NDArrayIndex.interval(0, trainSize),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );

            INDArray testFeatures = features.get(
                    NDArrayIndex.interval(trainSize, totalSamples),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
            INDArray testLabels = labels.get(
                    NDArrayIndex.interval(trainSize, totalSamples),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );

            logger.info("Data split - Train: {} samples, Test: {} samples",
                    trainSize, totalSamples - trainSize);

            return new SplitResult(trainFeatures, trainLabels, testFeatures, testLabels);
        } else if (features.rank() == 2 && labels.rank() == 2) {
            // 如果都是2D数组（样本数, 特征数）
            INDArray trainFeatures = features.get(
                    NDArrayIndex.interval(0, trainSize),
                    NDArrayIndex.all()
            );
            INDArray trainLabels = labels.get(
                    NDArrayIndex.interval(0, trainSize),
                    NDArrayIndex.all()
            );

            INDArray testFeatures = features.get(
                    NDArrayIndex.interval(trainSize, totalSamples),
                    NDArrayIndex.all()
            );
            INDArray testLabels = labels.get(
                    NDArrayIndex.interval(trainSize, totalSamples),
                    NDArrayIndex.all()
            );

            logger.info("Data split - Train: {} samples, Test: {} samples",
                    trainSize, totalSamples - trainSize);

            return new SplitResult(trainFeatures, trainLabels, testFeatures, testLabels);
        } else {
            throw new IllegalArgumentException("Unsupported array dimensions: features rank=" +
                    features.rank() + ", labels rank=" + labels.rank());
        }
    }

    /**
     * 评估模型
     */
    private EvaluationResult evaluateModel(MultiLayerNetwork model, SplitResult splitResult) {
        // 训练集评估
        INDArray trainPredictions = model.output(splitResult.trainFeatures);
        INDArray trainActual = splitResult.trainLabels;

        // 测试集评估
        INDArray testPredictions = model.output(splitResult.testFeatures);
        INDArray testActual = splitResult.testLabels;

        // 反标准化预测结果
        // 注意：实际使用时需要根据保存的标准化器进行反标准化

        return new EvaluationResult(trainPredictions, trainActual, testPredictions, testActual);
    }

    /**
     * 交叉验证
     */
    /*
    public CrossValidationResult crossValidate(INDArray features, INDArray labels, int folds) {
        logger.info("Starting {}-fold cross validation", folds);

        int samplesPerFold = (int) features.size(0) / folds;
        List<Double> foldScores = new ArrayList<>();

        for (int fold = 0; fold < folds; fold++) {
            int start = fold * samplesPerFold;
            int end = (fold == folds - 1) ? (int) features.size(0) : (fold + 1) * samplesPerFold;

            // 创建验证集
            INDArray valFeatures = features.get(Nd4j.createIndexArray(start, end), Nd4j.all(), Nd4j.all());
            INDArray valLabels = labels.get(Nd4j.createIndexArray(start, end), Nd4j.all());

            // 创建训练集（排除验证集）
            INDArray trainFeatures = null;
            INDArray trainLabels = null;

            if (fold == 0) {
                trainFeatures = features.get(Nd4j.createIndexArray(end, (int) features.size(0)),
                        Nd4j.all(), Nd4j.all());
                trainLabels = labels.get(Nd4j.createIndexArray(end, (int) labels.size(0)), Nd4j.all());
            } else if (fold == folds - 1) {
                trainFeatures = features.get(Nd4j.createIndexArray(0, start), Nd4j.all(), Nd4j.all());
                trainLabels = labels.get(Nd4j.createIndexArray(0, start), Nd4j.all());
            } else {
                INDArray firstPartFeatures = features.get(Nd4j.createIndexArray(0, start),
                        Nd4j.all(), Nd4j.all());
                INDArray secondPartFeatures = features.get(Nd4j.createIndexArray(end, (int) features.size(0)),
                        Nd4j.all(), Nd4j.all());
                trainFeatures = Nd4j.concat(0, firstPartFeatures, secondPartFeatures);

                INDArray firstPartLabels = labels.get(Nd4j.createIndexArray(0, start), Nd4j.all());
                INDArray secondPartLabels = labels.get(Nd4j.createIndexArray(end, (int) labels.size(0)), Nd4j.all());
                trainLabels = Nd4j.concat(0, firstPartLabels, secondPartLabels);
            }

            // 训练和评估当前折
            DataSetIterator trainIterator = createDataSetIterator(trainFeatures, trainLabels, config.getBatchSize());
            DataSetIterator valIterator = createDataSetIterator(valFeatures, valLabels, config.getBatchSize());

            LSTMModel lstmModel = new LSTMModel(config);
            lstmModel.initialize((int) features.size(2), config.getPredictSteps());

            MultiLayerNetwork model = lstmModel.getModel();
            model.fit(trainIterator, config.getEpochs());

            // 计算验证集损失
            double valLoss = model.score(valIterator);
            foldScores.add(valLoss);

            logger.info("Fold {}/{} completed. Validation Loss: {}",
                    fold + 1, folds, valLoss);
        }

        // 计算平均得分
        double avgScore = foldScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double stdScore = calculateStd(foldScores, avgScore);

        logger.info("Cross validation completed. Average Score: {}, Std: {}", avgScore, stdScore);

        return new CrossValidationResult(avgScore, stdScore, foldScores);
    }*/
    public CrossValidationResult crossValidate(INDArray features, INDArray labels, int folds) {
        logger.info("Starting {}-fold cross validation", folds);

        int totalSamples = (int) features.size(0);
        int samplesPerFold = totalSamples / folds;
        List<Double> foldScores = new ArrayList<>();

        for (int fold = 0; fold < folds; fold++) {
            int start = fold * samplesPerFold;
            int end = (fold == folds - 1) ? totalSamples : (fold + 1) * samplesPerFold;

            // 根据数组维度选择正确的索引方式
            INDArray valFeatures, valLabels;
            if (features.rank() == 3) {
                valFeatures = features.get(
                        NDArrayIndex.interval(start, end),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()
                );
            } else {
                valFeatures = features.get(
                        NDArrayIndex.interval(start, end),
                        NDArrayIndex.all()
                );
            }

            if (labels.rank() == 3) {
                valLabels = labels.get(
                        NDArrayIndex.interval(start, end),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()
                );
            } else {
                valLabels = labels.get(
                        NDArrayIndex.interval(start, end),
                        NDArrayIndex.all()
                );
            }

            // 创建训练集（排除验证集）
            INDArray trainFeatures = null;
            INDArray trainLabels = null;

            if (fold == 0) {
                // 第一折：验证集在前，训练集在后
                trainFeatures = getRows(features, end, totalSamples);
                trainLabels = getRows(labels, end, totalSamples);
            } else if (fold == folds - 1) {
                // 最后一折：验证集在后，训练集在前
                trainFeatures = getRows(features, 0, start);
                trainLabels = getRows(labels, 0, start);
            } else {
                // 中间折：验证集在中间，训练集为前后两部分
                // 第一部分：0 到 start
                INDArray firstPartFeatures = getRows(features, 0, start);
                INDArray firstPartLabels = getRows(labels, 0, start);

                // 第二部分：end 到 totalSamples
                INDArray secondPartFeatures = getRows(features, end, totalSamples);
                INDArray secondPartLabels = getRows(labels, end, totalSamples);

                // 合并两部分
                trainFeatures = Nd4j.concat(0, firstPartFeatures, secondPartFeatures);
                trainLabels = Nd4j.concat(0, firstPartLabels, secondPartLabels);
            }

            // 训练和评估当前折
            DataSetIterator trainIterator = createDataSetIterator(trainFeatures, trainLabels, config.getBatchSize());

            // 创建验证集 DataSet（而不是 Iterator）
            DataSet validationDataSet = new DataSet(valFeatures, valLabels);

            LSTMModel lstmModel = new LSTMModel(config);
            lstmModel.initialize((int) features.size(2), config.getPredictSteps());

            MultiLayerNetwork model = lstmModel.getModel();
            model.fit(trainIterator, config.getEpochs());

            // 计算验证集损失 - 直接使用 DataSet
            double valLoss = model.score(validationDataSet);
            foldScores.add(valLoss);

            logger.info("Fold {}/{} completed. Validation Loss: {}",
                    fold + 1, folds, valLoss);
        }

        // 计算平均得分
        double avgScore = foldScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double stdScore = calculateStd(foldScores, avgScore);

        logger.info("Cross validation completed. Average Score: {}, Std: {}", avgScore, stdScore);

        return new CrossValidationResult(avgScore, stdScore, foldScores);
    }

    // 辅助方法：获取指定行范围（处理不同维度的数组）
    private INDArray getRows(INDArray array, int start, int end) {
        if (start >= end) {
            // 返回空数组
            long[] shape = array.shape();
            shape[0] = 0; // 行数为0
            return Nd4j.create(shape);
        }

        int rank = array.rank();
        INDArrayIndex[] indices = new INDArrayIndex[rank]; // 使用 INDArrayIndex 接口类型

        // 第一维：行范围
        indices[0] = NDArrayIndex.interval(start, end);

        // 其余维度：全部
        for (int i = 1; i < rank; i++) {
            indices[i] = NDArrayIndex.all();
        }

        return array.get(indices);
    }

    // 计算标准差
    private double calculateStd(List<Double> values, double mean) {
        if (values.size() <= 1) {
            return 0.0;
        }

        double sum = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sum += diff * diff;
        }

        return Math.sqrt(sum / (values.size() - 1));
    }
    /*
    private double calculateStd(List<Double> values, double mean) {
        double variance = values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0.0);
        return Math.sqrt(variance);
    }*/

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
