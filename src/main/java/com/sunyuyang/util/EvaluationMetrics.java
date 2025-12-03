package com.sunyuyang.util;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EvaluationMetrics {

    /**
     * 计算均方误差 (MSE)
     */
    public static double calculateMSE(INDArray predictions, INDArray actuals) {
        INDArray diff = predictions.sub(actuals);
        INDArray squared = diff.mul(diff);
        return squared.meanNumber().doubleValue();
    }

    /**
     * 计算平均绝对误差 (MAE)
     */
    public static double calculateMAE(INDArray predictions, INDArray actuals) {
        INDArray diff = predictions.sub(actuals);
        INDArray abs = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.same.Abs(diff)).z();
        //INDArray abs = diff.dup().abs();
        return abs.meanNumber().doubleValue();
    }

    /**
     * 计算均方根误差 (RMSE)
     */
    public static double calculateRMSE(INDArray predictions, INDArray actuals) {
        return Math.sqrt(calculateMSE(predictions, actuals));
    }

    /**
     * 计算平均绝对百分比误差 (MAPE)
     */
    public static double calculateMAPE(INDArray predictions, INDArray actuals) {
        INDArray diff = predictions.sub(actuals);
        INDArray absDiff = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.same.Abs(diff)).z();
        INDArray absActuals = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.same.Abs(actuals)).z();

        // 避免除以零
        INDArray safeActuals = absActuals.add(1e-10);
        INDArray percentageErrors = absDiff.div(safeActuals).mul(100);

        return percentageErrors.meanNumber().doubleValue();
    }

    /**
     * 计算决定系数 (R²)
     */
    public static double calculateR2(INDArray predictions, INDArray actuals) {
        double actualMean = actuals.meanNumber().doubleValue();
        INDArray ssTotal = actuals.sub(actualMean).mul(actuals.sub(actualMean));
        double totalSum = ssTotal.sumNumber().doubleValue();

        INDArray ssResidual = predictions.sub(actuals).mul(predictions.sub(actuals));
        double residualSum = ssResidual.sumNumber().doubleValue();

        return 1 - (residualSum / (totalSum + 1e-10));
    }

    /**
     * 计算方向准确性 (Directional Accuracy)
     */
    public static double calculateDirectionalAccuracy(INDArray predictions, INDArray actuals) {
        if (predictions.size(0) < 2) {
            return 0.0;
        }

        int correctDirections = 0;
        int totalComparisons = 0;

        for (int i = 1; i < predictions.size(0); i++) {
            double predDirection = predictions.getDouble(i) - predictions.getDouble(i - 1);
            double actualDirection = actuals.getDouble(i) - actuals.getDouble(i - 1);

            // 如果方向相同（都正或都负）
            if ((predDirection >= 0 && actualDirection >= 0) ||
                    (predDirection < 0 && actualDirection < 0)) {
                correctDirections++;
            }

            totalComparisons++;
        }

        return totalComparisons > 0 ? (double) correctDirections / totalComparisons : 0.0;
    }

    /**
     * 打印所有评估指标
     */
    public static void printAllMetrics(INDArray predictions, INDArray actuals, String datasetName) {
        System.out.println("\n=== 模型评估结果 (" + datasetName + ") ===");
        System.out.printf("均方误差 (MSE): %.6f%n", calculateMSE(predictions, actuals));
        System.out.printf("平均绝对误差 (MAE): %.6f%n", calculateMAE(predictions, actuals));
        System.out.printf("均方根误差 (RMSE): %.6f%n", calculateRMSE(predictions, actuals));
        System.out.printf("平均绝对百分比误差 (MAPE): %.2f%%%n", calculateMAPE(predictions, actuals));
        System.out.printf("决定系数 (R²): %.4f%n", calculateR2(predictions, actuals));
        System.out.printf("方向准确性: %.2f%%%n", calculateDirectionalAccuracy(predictions, actuals) * 100);
    }
}
