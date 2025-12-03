package com.sunyuyang.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.util.NoSuchElementException;

public class StockPriceIterator implements DataSetIterator {
    private final INDArray features;
    private final INDArray labels;
    private final int batchSize;
    private final int totalExamples;
    private final int numFeatures;
    private final int numLabels;

    private int cursor = 0;
    private DataSetPreProcessor preProcessor;

    public StockPriceIterator(INDArray features, INDArray labels, int batchSize) {
        if (features.size(0) != labels.size(0)) {
            throw new IllegalArgumentException("Features and labels must have same number of examples");
        }

        this.features = features;
        this.labels = labels;
        this.batchSize = batchSize;
        this.totalExamples = (int) features.size(0);
        this.numFeatures = (int) features.size(1);
        this.numLabels = (int) labels.size(1);
    }

    /**
     * 创建滑动窗口数据集
     */
    public static StockPriceIterator createSlidingWindowDataset(INDArray timeSeriesData, int timeSteps,
                                                                int predictSteps, int batchSize) {
        int totalSamples = (int) timeSeriesData.size(0) - timeSteps - predictSteps + 1;

        if (totalSamples <= 0) {
            throw new IllegalArgumentException("Not enough data for given timeSteps and predictSteps");
        }

        int numFeatures = (int) timeSeriesData.size(1);

        INDArray features = Nd4j.create(totalSamples, timeSteps, numFeatures);
        INDArray labels = Nd4j.create(totalSamples, predictSteps);

        for (int i = 0; i < totalSamples; i++) {
            // 特征：从i到i+timeSteps-1的时间步
            INDArray featureWindow = timeSeriesData.get(
                    NDArrayIndex.interval(i, i + timeSteps),
                    NDArrayIndex.all()
            );

            // 标签：从i+timeSteps到i+timeSteps+predictSteps-1的收盘价（假设第一列是收盘价）
            INDArray labelWindow = timeSeriesData.get(
                    NDArrayIndex.interval(i + timeSteps, i + timeSteps + predictSteps),
                    NDArrayIndex.point(0) // 假设第一列是收盘价
            );

            features.put(new int[]{i, 0, 0}, featureWindow);
            labels.putRow(i, labelWindow.reshape(1, predictSteps));
        }

        return new StockPriceIterator(features.reshape(totalSamples, timeSteps * numFeatures),
                labels, batchSize);
    }

    @Override
    public DataSet next(int num) {
        if (cursor >= totalExamples) {
            throw new NoSuchElementException();
        }

        int actualNum = Math.min(num, totalExamples - cursor);
        int end = cursor + actualNum;

        INDArray featureBatch = features.get(NDArrayIndex.interval(cursor, end), NDArrayIndex.all());
        INDArray labelBatch = labels.get(NDArrayIndex.interval(cursor, end), NDArrayIndex.all());

        cursor = end;

        DataSet dataSet = new DataSet(featureBatch, labelBatch);

        if (preProcessor != null) {
            preProcessor.preProcess(dataSet);
        }

        return dataSet;
    }

    @Override
    public int inputColumns() {
        return numFeatures;
    }

    @Override
    public int totalOutcomes() {
        return numLabels;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null; // 回归问题，不需要类别标签
    }

    @Override
    public boolean hasNext() {
        return cursor < totalExamples;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
