package com.sunyuyang.entity;

public class ModelConfig {
    private int timeSteps = 60;          // 时间步长（历史窗口）
    private int predictSteps = 32;       // 预测步长（未来5个交易日）
    private int batchSize = 64;          // 批大小
    private int epochs = 100;            // 训练轮数
    private double trainTestSplit = 0.8; // 训练测试分割比例
    private int lstmLayer1Size = 128;    // 第一LSTM层大小
    private int lstmLayer2Size = 64;     // 第二LSTM层大小
    private int denseLayerSize = 32;     // 全连接层大小
    private double learningRate = 0.001; // 学习率
    private double dropoutRate = 0.2;    // Dropout率

    private String modelVariant = "default"; // 默认使用标准模型

    public String getModelVariant() {
        return modelVariant;
    }

    public void setModelVariant(String modelVariant) {
        this.modelVariant = modelVariant;
    }

    public static ModelConfig getDefaultConfig() {
        return new ModelConfig();
    }

    // Getters and Setters
    public int getTimeSteps() {
        return timeSteps;
    }

    public void setTimeSteps(int timeSteps) {
        this.timeSteps = timeSteps;
    }

    public int getPredictSteps() {
        return predictSteps;
    }

    public void setPredictSteps(int predictSteps) {
        this.predictSteps = predictSteps;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public double getTrainTestSplit() {
        return trainTestSplit;
    }

    public void setTrainTestSplit(double trainTestSplit) {
        this.trainTestSplit = trainTestSplit;
    }

    public int getLstmLayer1Size() {
        return lstmLayer1Size;
    }

    public void setLstmLayer1Size(int lstmLayer1Size) {
        this.lstmLayer1Size = lstmLayer1Size;
    }

    public int getLstmLayer2Size() {
        return lstmLayer2Size;
    }

    public void setLstmLayer2Size(int lstmLayer2Size) {
        this.lstmLayer2Size = lstmLayer2Size;
    }

    public int getDenseLayerSize() {
        return denseLayerSize;
    }

    public void setDenseLayerSize(int denseLayerSize) {
        this.denseLayerSize = denseLayerSize;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getDropoutRate() {
        return dropoutRate;
    }

    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }
}
