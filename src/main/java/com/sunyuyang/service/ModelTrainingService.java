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
import java.util.List;

public class ModelTrainingService {
    private static final Logger logger = LoggerFactory.getLogger(ModelTrainingService.class);
    private final ModelConfig config;

    public ModelTrainingService(ModelConfig config) {
        this.config = config;
    }

    /**
     * 训练模型 - 修复版本
     */
    public TrainingResult trainModel(INDArray features, INDArray labels, String modelName) {
        logger.info("Starting model training...");
        logger.info("Features shape: {}, Labels shape: {}", features.shape(), labels.shape());

        try {
            // 1. 检查数据有效性
            if (features.size(0) == 0 || labels.size(0) == 0) {
                throw new IllegalArgumentException("训练数据为空");
            }

            if (features.size(0) != labels.size(0)) {
                logger.warn("特征和标签样本数不匹配: {} vs {}", features.size(0), labels.size(0));
                // 取最小样本数
                int minSamples = (int) Math.min(features.size(0), labels.size(0));
                features = features.get(NDArrayIndex.interval(0, minSamples), NDArrayIndex.all());
                labels = labels.get(NDArrayIndex.interval(0, minSamples), NDArrayIndex.all());
                logger.info("调整后: 特征shape={}, 标签shape={}", features.shape(), labels.shape());
            }

            // 2. 划分训练集和测试集
            SplitResult splitResult = splitTrainTestSafe(features, labels, config.getTrainTestSplit());

            // 3. 创建数据迭代器
            DataSetIterator trainIterator = createDataSetIteratorFixed(
                    splitResult.trainFeatures, splitResult.trainLabels, config.getBatchSize());

            DataSetIterator testIterator = createDataSetIteratorFixed(
                    splitResult.testFeatures, splitResult.testLabels, config.getBatchSize());

            // 4. 初始化模型 - 获取正确的输入特征数
            int numInputFeatures;
            if (features.rank() == 3) {
                // 3D数组: [样本数, 时间步长, 特征数]
                numInputFeatures = (int) features.size(2);
                logger.info("使用3D特征，输入特征数: {}", numInputFeatures);
            } else {
                // 2D数组: [样本数, 特征数]
                numInputFeatures = (int) features.size(1);
                logger.info("使用2D特征，输入特征数: {}", numInputFeatures);
            }

            LSTMModel lstmModel = new LSTMModel(config);
            lstmModel.initialize(numInputFeatures, config.getPredictSteps());

            MultiLayerNetwork model = lstmModel.getModel();

            // 5. 简化训练（不使用早停，直接训练）
            logger.info("开始模型训练...");
            int actualEpochs = Math.min(config.getEpochs(), 50); // 限制最大训练轮次
            for (int epoch = 0; epoch < actualEpochs; epoch++) {
                model.fit(trainIterator);
                trainIterator.reset();

                if ((epoch + 1) % 5 == 0) {
                    double score = model.score();
                    logger.info("训练轮次 {}/{} - 损失: {:.6f}",
                            epoch + 1, actualEpochs, score);
                }
            }

            // 6. 评估模型
            EvaluationResult evalResult = evaluateModel(model, splitResult);

            // 7. 保存最终模型
            lstmModel.saveModel(modelName);

            logger.info("模型训练完成");

            return new TrainingResult(model, evalResult, null);

        } catch (Exception e) {
            logger.error("模型训练失败", e);
            throw new RuntimeException("模型训练失败", e);
        }
    }

    /**
     * 安全的划分训练集和测试集 - 修复数组索引问题
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

        // 使用安全的索引获取方法
        INDArray trainFeatures = getSafeRows(features, 0, trainSize);
        INDArray testFeatures = getSafeRows(features, trainSize, totalSamples);

        INDArray trainLabels = getSafeRows(labels, 0, trainSize);
        INDArray testLabels = getSafeRows(labels, trainSize, totalSamples);

        // 验证形状匹配
        logger.info("训练特征形状: {}", trainFeatures.shape());
        logger.info("训练标签形状: {}", trainLabels.shape());
        logger.info("测试特征形状: {}", testFeatures.shape());
        logger.info("测试标签形状: {}", testLabels.shape());

        return new SplitResult(trainFeatures, trainLabels, testFeatures, testLabels);
    }

    /**
     * 安全获取行数据 - 确保索引不超出范围
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
     * 创建数据集迭代器 - 修复版本，正确处理3D数据
     */
    private DataSetIterator createDataSetIteratorFixed(INDArray features, INDArray labels, int batchSize) {
        List<DataSet> dataSets = new ArrayList<>();

        if (features.size(0) == 0 || labels.size(0) == 0) {
            logger.warn("创建迭代器时发现空数据集");
            return new ListDataSetIterator<>(dataSets, batchSize);
        }

        int numSamples = (int) features.size(0);

        for (int i = 0; i < numSamples; i++) {
            // 特征处理：调整维度顺序
            INDArray feature;
            if (features.rank() == 3) {
                // 3D数组: [样本数, 时间步长, 特征数]
                // 获取单个样本: [timeSteps, features]
                feature = features.get(
                        NDArrayIndex.point(i),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()
                );

                // Deeplearning4j LSTM期望的顺序是: [batchSize, features, timeSteps]
                // 所以我们需要转置: [timeSteps, features] -> [features, timeSteps]
                feature = feature.transpose();

                // 添加batch维度: [features, timeSteps] -> [1, features, timeSteps]
                feature = feature.reshape(1, (int)feature.size(0), (int)feature.size(1));

                logger.debug("第{}个样本特征形状(调整后): {}", i, feature.shape());
            } else {
                throw new IllegalArgumentException("不支持的特征数组维度: rank=" + features.rank());
            }

            // 关键修复：标签处理
            INDArray label;
            if (labels.rank() == 3) {
                // 3D标签: [样本数, 预测步长, 1] 或 [样本数, 时间步长, 预测步长]
                label = labels.get(
                        NDArrayIndex.point(i),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()
                );

                // 简化：如果是3D，取最后一个时间步
                if (label.size(1) > 1) {
                    // 取最后一个时间步 [batchSize, 1, predictSteps]
                    label = label.get(
                            NDArrayIndex.point(0),
                            NDArrayIndex.point((int)label.size(1) - 1),
                            NDArrayIndex.all()
                    );
                }
            } else if (labels.rank() == 2) {
                // 2D标签: [样本数, 预测步长] - 最常见情况
                label = labels.get(
                        NDArrayIndex.point(i),
                        NDArrayIndex.all()
                );

                // 关键修复：LSTM输出层期望的标签形状
                // 对于单输出时间序列预测，标签应该是 [batchSize, predictSteps, 1]
                // 但我们也可以使用 [batchSize, predictSteps] 并让模型自动处理

                // 方法1：保持为2D，但在训练时注意
                // 重塑为 [1, predictSteps]
                label = label.reshape(1, label.size(0));

                // 方法2：或者转换为3D [1, 1, predictSteps]
                // label = label.reshape(1, 1, label.size(0));
            } else {
                throw new IllegalArgumentException("不支持的标签数组维度: rank=" + labels.rank());
            }

            logger.debug("第{}个样本标签形状: {}", i, label.shape());
            dataSets.add(new DataSet(feature, label));

            // 调试：打印第一个样本的形状
            if (i == 0) {
                logger.info("第一个样本 - 特征形状: {}, 标签形状: {}", feature.shape(), label.shape());
            }
        }

        int actualBatchSize = Math.min(batchSize, dataSets.size());
        if (actualBatchSize == 0) {
            actualBatchSize = 1;
        }

        logger.info("创建数据迭代器: 样本数={}, batch大小={}", dataSets.size(), actualBatchSize);
        return new ListDataSetIterator<>(dataSets, actualBatchSize);
    }

    /**
     * 评估模型
     */
    private EvaluationResult evaluateModel(MultiLayerNetwork model, SplitResult splitResult) {
        try {
            // 训练集评估 - 关键修复：确保输入是3D
            INDArray trainFeatures3D = ensure3D(splitResult.trainFeatures);
            INDArray trainPredictions = model.output(trainFeatures3D);
            INDArray trainActual = splitResult.trainLabels;

            // 测试集评估
            INDArray testFeatures3D = ensure3D(splitResult.testFeatures);
            INDArray testPredictions = model.output(testFeatures3D);
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
     * 确保输入是3D形状 [batchSize, timeSteps, features]
     */
    private INDArray ensure3D(INDArray input) {
        if (input.rank() == 3) {
            return input;
        } else if (input.rank() == 2) {
            // 如果是2D，添加时间步维度 [batchSize, 1, features]
            return input.reshape(input.size(0), 1, input.size(1));
        } else {
            throw new IllegalArgumentException("无法处理输入维度: rank=" + input.rank());
        }
    }

    /**
     * 简化版交叉验证
     */
    public CrossValidationResult crossValidate(INDArray features, INDArray labels, int folds) {
        logger.info("Starting {}-fold cross validation", folds);

        int totalSamples = (int) features.size(0);
        if (totalSamples < folds) {
            logger.warn("样本数少于折数，减少折数");
            folds = Math.min(totalSamples, 2);
        }

        int samplesPerFold = totalSamples / folds;
        List<Double> foldScores = new ArrayList<>();

        logger.info("总样本: {}, 每折样本: {}", totalSamples, samplesPerFold);
        logger.info("特征形状: {}, 标签形状: {}", features.shape(), labels.shape());

        for (int fold = 0; fold < folds; fold++) {
            int start = fold * samplesPerFold;
            int end = (fold == folds - 1) ? totalSamples : (fold + 1) * samplesPerFold;

            logger.info("第 {}/{} 折: 样本 {}-{}", fold + 1, folds, start, end);

            // 获取验证集
            INDArray valFeatures = getSafeRows(features, start, end);
            INDArray valLabels = getSafeRows(labels, start, end);

            // 创建训练集（排除验证集）
            INDArray trainFeatures, trainLabels;

            if (fold == 0) {
                // 第一折：验证集在前，训练集在后
                trainFeatures = getSafeRows(features, end, totalSamples);
                trainLabels = getSafeRows(labels, end, totalSamples);
            } else if (fold == folds - 1) {
                // 最后一折：验证集在后，训练集在前
                trainFeatures = getSafeRows(features, 0, start);
                trainLabels = getSafeRows(labels, 0, start);
            } else {
                // 中间折：验证集在中间，训练集为前后两部分
                INDArray firstPartFeatures = getSafeRows(features, 0, start);
                INDArray firstPartLabels = getSafeRows(labels, 0, start);

                INDArray secondPartFeatures = getSafeRows(features, end, totalSamples);
                INDArray secondPartLabels = getSafeRows(labels, end, totalSamples);

                // 合并两部分
                trainFeatures = Nd4j.concat(0, firstPartFeatures, secondPartFeatures);
                trainLabels = Nd4j.concat(0, firstPartLabels, secondPartLabels);
            }

            // 检查数据有效性
            if (trainFeatures.size(0) == 0 || trainLabels.size(0) == 0) {
                logger.warn("第 {} 折训练集为空，跳过", fold + 1);
                continue;
            }

            try {
                DataSetIterator trainIterator = createDataSetIteratorFixed(trainFeatures, trainLabels, config.getBatchSize());

                // 创建验证集
                DataSet validationDataSet = new DataSet(ensure3D(valFeatures), valLabels);

                // 重新初始化模型用于当前折
                LSTMModel lstmModel = new LSTMModel(config);

                // 获取正确的输入特征数
                int numInputFeatures;
                if (features.rank() == 3) {
                    numInputFeatures = (int) features.size(2);
                } else {
                    numInputFeatures = (int) features.size(1);
                }

                lstmModel.initialize(numInputFeatures, config.getPredictSteps());

                MultiLayerNetwork model = lstmModel.getModel();

                // 简化训练：只训练少量轮次
                int epochsPerFold = Math.min(5, config.getEpochs());
                for (int epoch = 0; epoch < epochsPerFold; epoch++) {
                    model.fit(trainIterator);
                    trainIterator.reset();
                }

                // 计算验证集损失
                double valLoss = model.score(validationDataSet);
                foldScores.add(valLoss);

                logger.info("第 {}/{} 折完成. 验证损失: {}", fold + 1, folds, valLoss);

            } catch (Exception e) {
                logger.error("第 {} 折错误", fold + 1, e);
                // 继续下一折
            }
        }

        if (foldScores.isEmpty()) {
            logger.error("所有折都失败!");
            return new CrossValidationResult(Double.NaN, Double.NaN, foldScores);
        }

        // 计算平均得分
        double avgScore = foldScores.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
        double stdScore = calculateStd(foldScores, avgScore);

        logger.info("交叉验证完成. 平均得分: {}, 标准差: {}", avgScore, stdScore);

        return new CrossValidationResult(avgScore, stdScore, foldScores);
    }

    /**
     * 计算标准差
     */
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

    /**
     * 创建数据集迭代器 - 简化版本
     */
    private DataSetIterator createSimpleDataSetIterator(INDArray features, INDArray labels, int batchSize) {
        List<DataSet> dataSets = new ArrayList<>();

        if (features.size(0) == 0 || labels.size(0) == 0) {
            logger.warn("创建迭代器时发现空数据集");
            return new ListDataSetIterator<>(dataSets, batchSize);
        }

        int numSamples = (int) features.size(0);

        for (int i = 0; i < numSamples; i++) {
            INDArray feature = features.get(
                    NDArrayIndex.point(i),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );

            INDArray label = labels.get(
                    NDArrayIndex.point(i),
                    NDArrayIndex.all()
            );

            dataSets.add(new DataSet(feature, label));
        }

        int actualBatchSize = Math.min(batchSize, dataSets.size());
        if (actualBatchSize == 0) {
            actualBatchSize = 1;
        }

        logger.info("创建数据迭代器: 样本数={}, batch大小={}", dataSets.size(), actualBatchSize);
        return new ListDataSetIterator<>(dataSets, actualBatchSize);
    }
}