package com.sunyuyang.util;

import org.nd4j.linalg.api.ndarray.INDArray;

public class PredictionDenormalizer {
    private final INDArray minValues;
    private final INDArray maxValues;
    private final double targetMin;
    private final double targetMax;

    public PredictionDenormalizer(INDArray originalData) {
        this.minValues = originalData.min(0);
        this.maxValues = originalData.max(0);
        this.targetMin = 0.0;
        this.targetMax = 1.0;
    }

    public PredictionDenormalizer(INDArray originalData, double targetMin, double targetMax) {
        this.minValues = originalData.min(0);
        this.maxValues = originalData.max(0);
        this.targetMin = targetMin;
        this.targetMax = targetMax;
    }

    public INDArray denormalize(INDArray normalizedPredictions) {
        // 反标准化公式: original = min + (normalized - targetMin) * (max - min) / (targetMax - targetMin)
        INDArray range = maxValues.sub(minValues);
        double targetRange = targetMax - targetMin;

        return normalizedPredictions.sub(targetMin)
                .muli(range)
                .divi(targetRange)
                .addi(minValues);
    }
}
