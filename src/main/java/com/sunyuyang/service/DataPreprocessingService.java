package com.sunyuyang.service;

import com.sunyuyang.entity.ZhituStockKLine;
import com.sunyuyang.feature.FeatureEngineeringService;
import com.sunyuyang.util.MinMaxNormalizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class DataPreprocessingService {
    private static final Logger logger = LoggerFactory.getLogger(DataPreprocessingService.class);
    private final FeatureEngineeringService featureEngineeringService;

    public DataPreprocessingService() {
        this.featureEngineeringService = new FeatureEngineeringService();
    }

    /**
     * 完整数据预处理流程
     */
    public ProcessedData preprocessData(List<ZhituStockKLine> klineData, int timeSteps, int predictSteps) {
        logger.info("Starting data preprocessing...");

        // 1. 特征工程
        INDArray basicFeatures = featureEngineeringService.extractBasicFeatures(klineData);
        INDArray enhancedFeatures = featureEngineeringService.addTechnicalIndicators(basicFeatures, klineData);
        INDArray cleanedFeatures = featureEngineeringService.handleMissingValues(enhancedFeatures);

        // 2. 创建标签
        INDArray labels = featureEngineeringService.createLabels(klineData, predictSteps);

        // 确保特征和标签数量匹配
        long minLengthLong = Math.min(cleanedFeatures.size(0), labels.size(0));
        int minLength = (int) minLengthLong;

        INDArray alignedFeatures = cleanedFeatures.get(
                NDArrayIndex.interval(0, minLength),
                NDArrayIndex.all()
        );

        INDArray alignedLabels = labels.get(
                NDArrayIndex.interval(0, minLength),
                NDArrayIndex.all()
        );

        // 3. 数据标准化
        MinMaxNormalizer featureNormalizer = new MinMaxNormalizer();
        featureNormalizer.fit(alignedFeatures);
        featureNormalizer.transform(alignedFeatures);

        MinMaxNormalizer labelNormalizer = new MinMaxNormalizer();
        labelNormalizer.fit(alignedLabels);
        labelNormalizer.transform(alignedLabels);

        // 4. 创建滑动窗口数据集 - 修正维度顺序
        // 确保输出维度为 [samples, features, time_steps] 而不是 [samples, time_steps, features]
        INDArray windowedFeatures = createSlidingWindows(alignedFeatures, timeSteps);
        INDArray windowedLabels = createLabelWindows(alignedLabels, timeSteps, predictSteps);

        logger.info("Data preprocessing completed. Features shape: {}, Labels shape: {}",
                windowedFeatures.shape(), windowedLabels.shape());

        return new ProcessedData(windowedFeatures, windowedLabels,
                featureNormalizer, labelNormalizer);
    }

    /**
     * 创建滑动窗口特征 - 修正维度顺序
     * 原版本：输出 [samples, time_steps, features]
     * 修正版：输出 [samples, features, time_steps]
     */
    private INDArray createSlidingWindows(INDArray features, int timeSteps) {
        int totalWindows = (int) features.size(0) - timeSteps + 1;
        int featureSize = (int) features.size(1);

        // 创建正确维度的数组：[totalWindows, featureSize, timeSteps]
        INDArray windowed = Nd4j.create(totalWindows, featureSize, timeSteps);

        for (int i = 0; i < totalWindows; i++) {
            // 获取时间窗口内的特征数据
            INDArray window = features.get(
                    NDArrayIndex.interval(i, i + timeSteps),
                    NDArrayIndex.all()
            );

            // 原始的window形状是 [timeSteps, featureSize]
            // 我们需要转置为 [featureSize, timeSteps] 然后放入三维数组
            // 首先转置窗口
            INDArray transposedWindow = window.transpose();

            // 将转置后的窗口放入三维数组的第i个切片
            windowed.putSlice(i, transposedWindow);
        }

        return windowed;
    }

    /**
     * 替代方案：创建滑动窗口特征（使用维度置换）
     * 如果上述方法有问题，可以使用这个替代方案
     */
    private INDArray createSlidingWindowsV2(INDArray features, int timeSteps) {
        int totalWindows = (int) features.size(0) - timeSteps + 1;
        int featureSize = (int) features.size(1);

        // 按照原方法创建 [totalWindows, timeSteps, featureSize]
        INDArray windowed = Nd4j.create(totalWindows, timeSteps, featureSize);

        for (int i = 0; i < totalWindows; i++) {
            INDArray window = features.get(
                    NDArrayIndex.interval(i, i + timeSteps),
                    NDArrayIndex.all()
            );
            windowed.putSlice(i, window);
        }

        // 使用维度置换转换为 [totalWindows, featureSize, timeSteps]
        return windowed.permute(0, 2, 1);
    }

    /**
     * 创建对应的标签窗口
     */
    private INDArray createLabelWindows(INDArray labels, int timeSteps, int predictSteps) {
        int totalWindows = (int) labels.size(0) - timeSteps - predictSteps + 1;

        if (totalWindows <= 0) {
            // 如果窗口数量不足，创建一个空数组
            logger.warn("无法创建标签窗口，总窗口数: {}", totalWindows);
            return Nd4j.create(0, predictSteps);
        }

        INDArray windowedLabels = Nd4j.create(totalWindows, predictSteps);

        for (int i = 0; i < totalWindows; i++) {
            INDArray labelWindow = labels.getRow(i + timeSteps - 1);
            windowedLabels.putRow(i, labelWindow);
        }

        return windowedLabels;
    }

    /**
     * 保存标准化器
     */
    public void saveNormalizers(MinMaxNormalizer featureNormalizer,
                                MinMaxNormalizer labelNormalizer,
                                String modelName) {
        try {
            File modelDir = new File("models/" + modelName);
            if (!modelDir.exists()) {
                modelDir.mkdirs();
            }

            featureNormalizer.save(new File(modelDir, "feature_normalizer.bin"));
            labelNormalizer.save(new File(modelDir, "label_normalizer.bin"));

            logger.info("Normalizers saved successfully");
        } catch (IOException e) {
            logger.error("Failed to save normalizers", e);
        }
    }

    /**
     * 加载标准化器
     */
    public NormalizerPair loadNormalizers(String modelName) {
        try {
            File modelDir = new File("models/" + modelName);

            MinMaxNormalizer featureNormalizer = NormalizerSerializer.getDefault()
                    .restore(new File(modelDir, "feature_normalizer.bin"));
            MinMaxNormalizer labelNormalizer = NormalizerSerializer.getDefault()
                    .restore(new File(modelDir, "label_normalizer.bin"));

            logger.info("Normalizers loaded successfully");
            return new NormalizerPair(featureNormalizer, labelNormalizer);
        } catch (Exception e) {
            logger.error("Failed to load normalizers", e);
            return null;
        }
    }

    /**
     * 处理后的数据容器类
     */
    public static class ProcessedData {
        private final INDArray features;
        private final INDArray labels;
        private final MinMaxNormalizer featureNormalizer;
        private final MinMaxNormalizer labelNormalizer;

        public ProcessedData(INDArray features, INDArray labels,
                             MinMaxNormalizer featureNormalizer,
                             MinMaxNormalizer labelNormalizer) {
            this.features = features;
            this.labels = labels;
            this.featureNormalizer = featureNormalizer;
            this.labelNormalizer = labelNormalizer;
        }

        public INDArray getFeatures() {
            return features;
        }

        public INDArray getLabels() {
            return labels;
        }

        public MinMaxNormalizer getFeatureNormalizer() {
            return featureNormalizer;
        }

        public MinMaxNormalizer getLabelNormalizer() {
            return labelNormalizer;
        }
    }

    /**
     * 标准化器对
     */
    public static class NormalizerPair {
        private final MinMaxNormalizer featureNormalizer;
        private final MinMaxNormalizer labelNormalizer;

        public NormalizerPair(MinMaxNormalizer featureNormalizer,
                              MinMaxNormalizer labelNormalizer) {
            this.featureNormalizer = featureNormalizer;
            this.labelNormalizer = labelNormalizer;
        }

        public MinMaxNormalizer getFeatureNormalizer() {
            return featureNormalizer;
        }

        public MinMaxNormalizer getLabelNormalizer() {
            return labelNormalizer;
        }
    }
}