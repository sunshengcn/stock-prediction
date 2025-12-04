package com.sunyuyang.feature;

import com.sunyuyang.entity.ZhituStockKLine;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class FeatureEngineeringService {
    private static final Logger logger = LoggerFactory.getLogger(FeatureEngineeringService.class);

    /**
     * 提取基础特征
     */
    public INDArray extractBasicFeatures(List<ZhituStockKLine> klineData) {
        int timesteps = klineData.size();
        int features = 12; // 基础特征数量

        INDArray featureArray = Nd4j.create(timesteps, features);

        for (int i = 0; i < timesteps; i++) {
            ZhituStockKLine k = klineData.get(i);
            int featureIndex = 0;

            // 1. 价格特征
            featureArray.putScalar(i, featureIndex++, k.getOpen());
            featureArray.putScalar(i, featureIndex++, k.getHigh());
            featureArray.putScalar(i, featureIndex++, k.getLow());
            featureArray.putScalar(i, featureIndex++, k.getClose());

            // 2. 成交量特征
            featureArray.putScalar(i, featureIndex++, Math.log1p(k.getVolume())); // 对数成交量
            featureArray.putScalar(i, featureIndex++, k.getAmount());

            // 3. 价格变动特征
            if (i > 0) {
                ZhituStockKLine prevK = klineData.get(i - 1);
                double priceChange = (k.getClose() - prevK.getClose()) / prevK.getClose();
                featureArray.putScalar(i, featureIndex++, priceChange);

                double volumeChange = (k.getVolume() - prevK.getVolume()) / (prevK.getVolume() + 1);
                featureArray.putScalar(i, featureIndex++, volumeChange);
            } else {
                featureArray.putScalar(i, featureIndex++, 0.0);
                featureArray.putScalar(i, featureIndex++, 0.0);
            }

            // 4. 技术指标特征（简化版）
            // 价格区间
            featureArray.putScalar(i, featureIndex++, (k.getHigh() - k.getLow()) / k.getOpen());

            // 收盘价与前收盘价比
            featureArray.putScalar(i, featureIndex++, (k.getClose() - k.getPrevClose()) / k.getPrevClose());

            // 平均成交价
            double avgPrice = k.getVolume() > 0 ? k.getAmount() / k.getVolume() : k.getClose();
            featureArray.putScalar(i, featureIndex++, avgPrice);

            // 交易状态
            if (k.getIsSuspended().equals("0")) {
                featureArray.putScalar(i, featureIndex++, 0.0);
            } else {
                featureArray.putScalar(i, featureIndex++, 1.0);
            }

        }

        logger.info("Extracted {} basic features from {} timesteps", features, timesteps);
        return featureArray;
    }

    /**
     * 添加技术指标
     */
    public INDArray addTechnicalIndicators(INDArray basicFeatures, List<ZhituStockKLine> klineData) {
        int timesteps = (int) basicFeatures.size(0);
        int basicFeatureCount = (int) basicFeatures.size(1);
        int technicalIndicatorCount = 8; // 技术指标数量

        INDArray enhancedFeatures = Nd4j.create(timesteps, basicFeatureCount + technicalIndicatorCount);

        // 复制基础特征
       /* enhancedFeatures.put(
                Nd4j.arange(0, timesteps),
                Nd4j.arange(0, basicFeatureCount),
                basicFeatures
        );*/

        enhancedFeatures.get(
                NDArrayIndex.interval(0, timesteps),
                NDArrayIndex.interval(0, basicFeatureCount)
        ).assign(basicFeatures);

        // 计算技术指标
        for (int i = 0; i < timesteps; i++) {
            int featureIndex = basicFeatureCount;

            // 1. 简单移动平均 (SMA)
            if (i >= 4) { // 5周期SMA
                double sma5 = 0;
                for (int j = 0; j < 5; j++) {
                    sma5 += klineData.get(i - j).getClose();
                }
                enhancedFeatures.putScalar(i, featureIndex++, sma5 / 5);
            } else {
                enhancedFeatures.putScalar(i, featureIndex++, klineData.get(i).getClose());
            }

            // 2. 指数移动平均 (EMA)
            if (i > 0) {
                double close = klineData.get(i).getClose();
                double prevEma = enhancedFeatures.getDouble(i - 1, featureIndex);
                double alpha = 2.0 / (12 + 1); // 12周期EMA的alpha
                double ema = alpha * close + (1 - alpha) * prevEma;
                enhancedFeatures.putScalar(i, featureIndex++, ema);
            } else {
                enhancedFeatures.putScalar(i, featureIndex++, klineData.get(i).getClose());
            }

            // 3. 相对强弱指数 (RSI) - 简化版
            if (i >= 14) {
                double avgGain = 0;
                double avgLoss = 0;

                for (int j = 1; j <= 14; j++) {
                    double change = klineData.get(i - j + 1).getClose() -
                            klineData.get(i - j).getClose();
                    if (change > 0) {
                        avgGain += change;
                    } else {
                        avgLoss -= change;
                    }
                }

                avgGain /= 14;
                avgLoss /= 14;

                double rs = avgLoss > 0 ? avgGain / avgLoss : 100;
                double rsi = 100 - (100 / (1 + rs));
                enhancedFeatures.putScalar(i, featureIndex++, rsi);
            } else {
                enhancedFeatures.putScalar(i, featureIndex++, 50.0); // 中性值
            }

            // 4. 布林带位置
            if (i >= 19) {
                double sum = 0;
                for (int j = 0; j < 20; j++) {
                    sum += klineData.get(i - j).getClose();
                }
                double middle = sum / 20;

                double variance = 0;
                for (int j = 0; j < 20; j++) {
                    double diff = klineData.get(i - j).getClose() - middle;
                    variance += diff * diff;
                }
                double stdDev = Math.sqrt(variance / 20);

                double bbPosition = (klineData.get(i).getClose() - middle) / (2 * stdDev);
                enhancedFeatures.putScalar(i, featureIndex++, bbPosition);
            } else {
                enhancedFeatures.putScalar(i, featureIndex++, 0.0);
            }

            // 5. 动量指标
            if (i > 9) {
                double momentum = klineData.get(i).getClose() - klineData.get(i - 10).getClose();
                enhancedFeatures.putScalar(i, featureIndex++, momentum / klineData.get(i - 10).getClose());
            } else {
                enhancedFeatures.putScalar(i, featureIndex++, 0.0);
            }

            // 6. 价格波动率
            if (i >= 9) {
                double high = Double.MIN_VALUE;
                double low = Double.MAX_VALUE;

                for (int j = 0; j < 10; j++) {
                    ZhituStockKLine k = klineData.get(i - j);
                    high = Math.max(high, k.getHigh());
                    low = Math.min(low, k.getLow());
                }

                double volatility = (high - low) / klineData.get(i).getClose();
                enhancedFeatures.putScalar(i, featureIndex++, volatility);
            } else {
                enhancedFeatures.putScalar(i, featureIndex++, 0.0);
            }

            // 7. 成交量加权平均价 (VWAP) - 简化版
            if (i >= 4) {
                double totalAmount = 0;
                double totalVolume = 0;

                for (int j = 0; j < 5; j++) {
                    ZhituStockKLine k = klineData.get(i - j);
                    totalAmount += k.getAmount();
                    totalVolume += k.getVolume();
                }

                double vwap = totalVolume > 0 ? totalAmount / totalVolume : klineData.get(i).getClose();
                enhancedFeatures.putScalar(i, featureIndex++, vwap);
            } else {
                enhancedFeatures.putScalar(i, featureIndex++, klineData.get(i).getClose());
            }

            // 8. 价格加速度
            if (i >= 2) {
                double acceleration = (klineData.get(i).getClose() - 2 * klineData.get(i - 1).getClose() +
                        klineData.get(i - 2).getClose()) / klineData.get(i - 2).getClose();
                enhancedFeatures.putScalar(i, featureIndex++, acceleration);
            } else {
                enhancedFeatures.putScalar(i, featureIndex++, 0.0);
            }
        }

        logger.info("Added {} technical indicators, total features: {}",
                technicalIndicatorCount, enhancedFeatures.size(1));
        return enhancedFeatures;
    }

    /**
     * 创建标签数据
     */
    public INDArray createLabels(List<ZhituStockKLine> klineData, int predictSteps) {
        int timesteps = klineData.size() - predictSteps;

        if (timesteps <= 0) {
            throw new IllegalArgumentException("Not enough data for prediction");
        }

        INDArray labelArray = Nd4j.create(timesteps, predictSteps);

        for (int i = 0; i < timesteps; i++) {
            for (int j = 0; j < predictSteps; j++) {
                // 归一化的价格变化作为标签
                double futurePrice = klineData.get(i + j + 1).getClose();
                double currentPrice = klineData.get(i).getClose();
                double priceChange = (futurePrice - currentPrice) / currentPrice;

                labelArray.putScalar(i, j, priceChange);
            }
        }

        logger.info("Created labels for {} timesteps, predicting {} steps ahead",
                timesteps, predictSteps);
        return labelArray;
    }

    /**
     * 处理缺失值和异常值
     */
    public INDArray handleMissingValues(INDArray features) {
        INDArray cleaned = features.dup();
        int rows = (int) cleaned.size(0);
        int cols = (int) cleaned.size(1);

        for (int col = 0; col < cols; col++) {
            // 计算列的平均值和标准差
            INDArray column = cleaned.getColumn(col);
            double mean = column.meanNumber().doubleValue();
            double std = column.stdNumber().doubleValue();

            // 识别和处理异常值（3σ原则）
            for (int row = 0; row < rows; row++) {
                double value = cleaned.getDouble(row, col);

                if (Double.isNaN(value) || Double.isInfinite(value)) {
                    // 用列平均值填充缺失值
                    cleaned.putScalar(row, col, mean);
                } else if (Math.abs(value - mean) > 3 * std) {
                    // 缩尾处理：将异常值调整到3σ范围内
                    double cappedValue = mean + Math.signum(value - mean) * 3 * std;
                    cleaned.putScalar(row, col, cappedValue);
                }
            }
        }

        return cleaned;
    }
}
