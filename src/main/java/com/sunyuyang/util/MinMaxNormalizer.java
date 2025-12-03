package com.sunyuyang.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * 最小-最大标准化器
 * 将数据缩放到指定的范围（默认为 [0, 1]）
 */
public class MinMaxNormalizer {
    private double dataMin;
    private double dataMax;
    private double targetMin;
    private double targetMax;
    private boolean fitted = false;

    /**
     * 默认构造函数，目标范围为 [0, 1]
     */
    public MinMaxNormalizer() {
        this(0.0, 1.0);
    }

    /**
     * 指定目标范围的构造函数
     *
     * @param targetMin 目标最小值
     * @param targetMax 目标最大值
     */
    public MinMaxNormalizer(double targetMin, double targetMax) {
        this.targetMin = targetMin;
        this.targetMax = targetMax;
    }

    /**
     * 创建适合列数据的标准化器（每列独立标准化）
     */
    public static ColumnMinMaxNormalizer createColumnNormalizer() {
        return new ColumnMinMaxNormalizer();
    }

    /**
     * 拟合数据，计算数据的最小值和最大值
     *
     * @param data 要拟合的数据
     */
    public void fit(INDArray data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("数据不能为空");
        }

        this.dataMin = Nd4j.min(data).getDouble(0);
        this.dataMax = Nd4j.max(data).getDouble(0);
        this.fitted = true;
    }

    /**
     * 拟合并转换数据（原地修改）
     *
     * @param data 要转换的数据
     */
    public void fitTransform(INDArray data) {
        fit(data);
        transform(data);
    }

    /**
     * 转换数据（原地修改）
     *
     * @param data 要转换的数据
     */
    public void transform(INDArray data) {
        if (!fitted) {
            throw new IllegalStateException("必须先调用 fit() 方法");
        }

        double dataRange = dataMax - dataMin;
        double targetRange = targetMax - targetMin;

        if (dataRange > 0) {
            // 标准化公式: targetMin + (x - dataMin) * (targetRange / dataRange)
            data.subi(dataMin)
                    .muli(targetRange / dataRange)
                    .addi(targetMin);
        } else {
            // 如果所有值相同，设为目标范围的中点
            data.assign((targetMin + targetMax) / 2.0);
        }
    }

    /**
     * 转换数据并返回新数组（不修改原始数据）
     *
     * @param data 要转换的数据
     * @return 转换后的数据
     */
    public INDArray transformCopy(INDArray data) {
        INDArray copy = data.dup();
        transform(copy);
        return copy;
    }

    /**
     * 反标准化（从目标范围恢复到原始范围）
     *
     * @param normalizedData 标准化后的数据
     */
    public void inverseTransform(INDArray normalizedData) {
        if (!fitted) {
            throw new IllegalStateException("必须先调用 fit() 方法");
        }

        double dataRange = dataMax - dataMin;
        double targetRange = targetMax - targetMin;

        if (dataRange > 0 && targetRange > 0) {
            // 反标准化公式: dataMin + (x - targetMin) * (dataRange / targetRange)
            normalizedData.subi(targetMin)
                    .muli(dataRange / targetRange)
                    .addi(dataMin);
        } else {
            // 如果范围无效，设为数据的最小值
            normalizedData.assign(dataMin);
        }
    }

    /**
     * 反标准化并返回新数组
     *
     * @param normalizedData 标准化后的数据
     * @return 原始范围的数据
     */
    public INDArray inverseTransformCopy(INDArray normalizedData) {
        INDArray copy = normalizedData.dup();
        inverseTransform(copy);
        return copy;
    }

    // 获取和设置方法
    public double getDataMin() {
        return dataMin;
    }

    public double getDataMax() {
        return dataMax;
    }

    public double getTargetMin() {
        return targetMin;
    }

    public double getTargetMax() {
        return targetMax;
    }

    public boolean isFitted() {
        return fitted;
    }

    /**
     * 列标准化器（每列独立标准化）
     */
    public static class ColumnMinMaxNormalizer {
        private INDArray columnMins;
        private INDArray columnMaxs;
        private double targetMin = 0.0;
        private double targetMax = 1.0;
        private boolean fitted = false;

        public void fit(INDArray data) {
            this.columnMins = data.min(0);  // 每列的最小值
            this.columnMaxs = data.max(0);  // 每列的最大值
            this.fitted = true;
        }

        public void transform(INDArray data) {
            if (!fitted) {
                throw new IllegalStateException("必须先调用 fit() 方法");
            }

            for (int col = 0; col < data.columns(); col++) {
                double dataMin = columnMins.getDouble(col);
                double dataMax = columnMaxs.getDouble(col);
                double dataRange = dataMax - dataMin;

                if (dataRange > 0) {
                    INDArray column = data.getColumn(col);
                    column.subi(dataMin)
                            .muli((targetMax - targetMin) / dataRange)
                            .addi(targetMin);
                } else {
                    // 如果该列所有值相同，设为目标范围的中点
                    data.getColumn(col).assign((targetMin + targetMax) / 2.0);
                }
            }
        }

        public void inverseTransform(INDArray normalizedData) {
            if (!fitted) {
                throw new IllegalStateException("必须先调用 fit() 方法");
            }

            for (int col = 0; col < normalizedData.columns(); col++) {
                double dataMin = columnMins.getDouble(col);
                double dataMax = columnMaxs.getDouble(col);
                double dataRange = dataMax - dataMin;
                double targetRange = targetMax - targetMin;

                if (dataRange > 0 && targetRange > 0) {
                    INDArray column = normalizedData.getColumn(col);
                    column.subi(targetMin)
                            .muli(dataRange / targetRange)
                            .addi(dataMin);
                } else {
                    normalizedData.getColumn(col).assign(dataMin);
                }
            }
        }

    }

    /**
     * 保存标准化器到文件
     */
    public void save(String filePath) throws IOException {
        save(new File(filePath));
    }

    public void save(File file) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(file))) {
            oos.writeObject(this);
        }
    }

    /**
     * 从文件加载标准化器
     */
    public static MinMaxNormalizer load(String filePath) throws IOException, ClassNotFoundException {
        return load(new File(filePath));
    }

    public static MinMaxNormalizer load(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(file))) {
            return (MinMaxNormalizer) ois.readObject();
        }
    }

    /**
     * 获取默认配置的标准化器（如果确实需要这个方法）
     */
    public static MinMaxNormalizer getDefault() {
        return new MinMaxNormalizer(0.0, 1.0);
    }
}
