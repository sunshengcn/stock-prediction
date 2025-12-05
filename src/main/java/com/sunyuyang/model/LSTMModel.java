package com.sunyuyang.model;

import com.sunyuyang.entity.ModelConfig;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class LSTMModel {
    private static final Logger logger = LoggerFactory.getLogger(LSTMModel.class);
    private MultiLayerNetwork model;
    private final ModelConfig config;

    public LSTMModel(ModelConfig config) {
        this.config = config;
    }

    /**
     * 加载模型
     */
    public static LSTMModel loadModel(String modelName) {
        try {
            File modelFile = new File("models/" + modelName + "/model.zip");

            if (!modelFile.exists()) {
                throw new IOException("Model file not found: " + modelFile.getPath());
            }

            MultiLayerNetwork loadedModel = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            LSTMModel lstmModel = new LSTMModel(ModelConfig.getDefaultConfig());
            lstmModel.model = loadedModel;

            logger.info("Model loaded from: {}", modelFile.getAbsolutePath());
            return lstmModel;
        } catch (IOException e) {
            logger.error("Failed to load model", e);
            throw new RuntimeException("Failed to load model", e);
        }
    }

    /**
     * 构建LSTM模型配置 - 修复版本：正确处理时间序列预测
     */
    public MultiLayerConfiguration buildModelConfig(int numInputFeatures, int numOutputSteps) {
        logger.info("Building LSTM model configuration...");
        logger.info("Input features: {}, Output steps: {}, Time steps: {}",
                numInputFeatures, numOutputSteps, config.getTimeSteps());

        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(config.getLearningRate()))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4) // L2正则化
                .list()

                // 第一LSTM层
                .layer(new LSTM.Builder()
                        .name("lstm-layer-1")
                        .nIn(numInputFeatures)  // 输入特征数
                        .nOut(config.getLstmLayer1Size())
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.SIGMOID)
                        .dropOut(config.getDropoutRate())
                        .build())

                // 第二LSTM层
                .layer(new LSTM.Builder()
                        .name("lstm-layer-2")
                        .nIn(config.getLstmLayer1Size())
                        .nOut(config.getLstmLayer2Size())
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.SIGMOID)
                        .dropOut(config.getDropoutRate())
                        .build())

                // 全连接层
                .layer(new DenseLayer.Builder()
                        .name("dense-layer")
                        .nIn(config.getLstmLayer2Size())
                        .nOut(config.getDenseLayerSize())
                        .activation(Activation.RELU)
                        .build())

                // 输出层
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .name("output-layer")
                        .nIn(config.getDenseLayerSize())
                        .nOut(numOutputSteps)
                        .activation(Activation.IDENTITY) // 回归问题使用线性激活
                        .build())

                // 设置输入类型（时间序列）
                .setInputType(InputType.recurrent(numInputFeatures))
                .build();
    }

    /**
     * 替代方案：使用简化模型配置（单层LSTM）
     */
    public MultiLayerConfiguration buildSimpleModelConfig(int numInputFeatures, int numOutputSteps) {
        logger.info("Building simple LSTM model configuration...");

        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(config.getLearningRate()))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()

                // 单层LSTM
                .layer(new LSTM.Builder()
                        .name("lstm-layer")
                        .nIn(numInputFeatures)
                        .nOut(64)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.SIGMOID)
                        .dropOut(config.getDropoutRate())
                        .build())

                // 输出层
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .name("output-layer")
                        .nIn(64)
                        .nOut(numOutputSteps)
                        .activation(Activation.IDENTITY)
                        .build())

                .setInputType(InputType.recurrent(numInputFeatures))
                .build();
    }

    /**
     * 初始化模型
     */
    public void initialize(int numInputFeatures, int numOutputSteps) {
        MultiLayerConfiguration config = buildModelConfig(numInputFeatures, numOutputSteps);

        // 如果你想使用简化模型，可以取消下面这行的注释：
        // MultiLayerConfiguration config = buildSimpleModelConfig(numInputFeatures, numOutputSteps);

        this.model = new MultiLayerNetwork(config);
        this.model.init();

        // 添加训练监听器
        this.model.setListeners(new ScoreIterationListener(100));

        logger.info("Model initialized with {} parameters", this.model.numParams());
        logger.info("Model summary:\n{}", getModelSummary());
    }

    /**
     * 获取模型
     */
    public MultiLayerNetwork getModel() {
        if (model == null) {
            throw new IllegalStateException("Model not initialized. Call initialize() first.");
        }
        return model;
    }

    /**
     * 保存模型
     */
    public void saveModel(String modelName) {
        try {
            File modelDir = new File("models/" + modelName);
            if (!modelDir.exists()) {
                modelDir.mkdirs();
            }

            File modelFile = new File(modelDir, "model.zip");
            ModelSerializer.writeModel(model, modelFile, true);

            logger.info("Model saved to: {}", modelFile.getAbsolutePath());
        } catch (IOException e) {
            logger.error("Failed to save model", e);
            throw new RuntimeException("Failed to save model", e);
        }
    }

    /**
     * 获取模型摘要
     */
    public String getModelSummary() {
        if (model == null) {
            return "Model not initialized";
        }

        StringBuilder summary = new StringBuilder();
        summary.append("=== LSTM Model Summary ===\n");
        summary.append(String.format("Total Parameters: %,d%n", model.numParams()));
        summary.append(String.format("Number of Layers: %d%n", model.getnLayers()));

        for (int i = 0; i < model.getnLayers(); i++) {
            org.deeplearning4j.nn.api.Layer layer = model.getLayer(i);
            summary.append(String.format("Layer %d: %s - %s%n",
                    i, layer.conf().getLayer().getLayerName(),
                    layer.type()));
        }

        return summary.toString();
    }

}