package com.sunyuyang.service;

import com.sunyuyang.entity.StockKLine;
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
    public ProcessedData preprocessData(List<StockKLine> klineData, int timeSteps, int predictSteps) {
        logger.info("Starting data preprocessing...");

        // 1. 特征工程
        INDArray basicFeatures = featureEngineeringService.extractBasicFeatures(klineData);
        INDArray enhancedFeatures = featureEngineeringService.addTechnicalIndicators(basicFeatures, klineData);
        INDArray cleanedFeatures = featureEngineeringService.handleMissingValues(enhancedFeatures);

        // 2. 创建标签
        INDArray labels = featureEngineeringService.createLabels(klineData, predictSteps);

        // 确保特征和标签数量匹配
        long minLengthLong = Math.min(cleanedFeatures.size(0), labels.size(0));
        // 使用时再转换为 int
        int minLength = (int) minLengthLong;
        //int minLength = Math.min(cleanedFeatures.size(0), labels.size(0));
        //INDArray alignedFeatures = cleanedFeatures.get(Nd4j.createIndexArray(0, minLength), Nd4j.all());
        //INDArray alignedLabels = labels.get(Nd4j.createIndexArray(0, minLength), Nd4j.all());

        // 正确的方式：
        INDArray alignedFeatures = cleanedFeatures.get(
                NDArrayIndex.interval(0, minLength),
                NDArrayIndex.all()
        );
        // 同样对于标签：
        INDArray alignedLabels = labels.get(
                NDArrayIndex.interval(0, minLength),
                NDArrayIndex.all()
        );

        // 3. 数据标准化
        //NormalizerMinMaxScaler featureNormalizer = new NormalizerMinMaxScaler();
        //featureNormalizer.fit(alignedFeatures);
        //featureNormalizer.transform(alignedFeatures);

        MinMaxNormalizer featureNormalizer = new MinMaxNormalizer();
        featureNormalizer.fit(alignedFeatures);
        featureNormalizer.transform(alignedFeatures);

        //NormalizerMinMaxScaler labelNormalizer = new NormalizerMinMaxScaler();
        //labelNormalizer.fit(alignedLabels);
        //labelNormalizer.transform(alignedLabels);

        MinMaxNormalizer labelNormalizer = new MinMaxNormalizer();
        labelNormalizer.fit(alignedLabels);
        labelNormalizer.transform(alignedLabels);

        // 4. 创建滑动窗口数据集
        INDArray windowedFeatures = createSlidingWindows(alignedFeatures, timeSteps);
        INDArray windowedLabels = createLabelWindows(alignedLabels, timeSteps, predictSteps);

        logger.info("Data preprocessing completed. Features shape: {}, Labels shape: {}",
                windowedFeatures.shape(), windowedLabels.shape());

        return new ProcessedData(windowedFeatures, windowedLabels,
                featureNormalizer, labelNormalizer);
    }

    /**
     * 创建滑动窗口特征
     */
    private INDArray createSlidingWindows(INDArray features, int timeSteps) {
        int totalWindows = (int) features.size(0) - timeSteps + 1;
        int featureSize = (int) features.size(1);

        INDArray windowed = Nd4j.create(totalWindows, timeSteps, featureSize);

        for (int i = 0; i < totalWindows; i++) {
            //INDArray window = features.get(
            //        Nd4j.createIndexArray(i, i + timeSteps),
            //        Nd4j.all()
            //);
            INDArray window = features.get(
                    NDArrayIndex.interval(i, i + timeSteps),
                    NDArrayIndex.all()
            );
            windowed.put(new int[]{i, 0, 0}, window);
        }

        return windowed;
    }

    /**
     * 创建对应的标签窗口
     */
    private INDArray createLabelWindows(INDArray labels, int timeSteps, int predictSteps) {
        int totalWindows = (int) labels.size(0) - timeSteps - predictSteps + 1;

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

            //MinMaxNormalizer.getDefault().write(featureNormalizer,
            //       new File(modelDir, "feature_normalizer.bin"));
            featureNormalizer.save(new File(modelDir, "feature_normalizer.bin"));
            //MinMaxNormalizer.getDefault().write(labelNormalizer,
            //        new File(modelDir, "label_normalizer.bin"));
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

            //MinMaxNormalizer featureNormalizer = MinMaxNormalizer.getDefault()
            //        .restore(new File(modelDir, "feature_normalizer.bin"));
            MinMaxNormalizer featureNormalizer = NormalizerSerializer.getDefault()
                    .restore(new File(modelDir, "feature_normalizer.bin"));
            //MinMaxNormalizer labelNormalizer = MinMaxNormalizer.getDefault()
            //        .restore(new File(modelDir, "label_normalizer.bin"));
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
