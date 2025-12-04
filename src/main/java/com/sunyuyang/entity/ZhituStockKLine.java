package com.sunyuyang.entity;

import java.time.LocalDateTime;

public class ZhituStockKLine {
    private String stockCode;//股票代码
    private String tradeTime;//交易时间
    private double open;//开盘价
    private double high;//最高价
    private double low;//最低价
    private double close;//收盘价
    private double volume;//成交量
    private double amount;//成交额
    private double prevClose;//前收盘价
    private String isSuspended;//停牌 1停牌，0 不停牌
    private String timeLevel;//K线类型：5：5分钟线、15：15分钟线、30：30分钟线、60：60分钟线、d：日线、w：周线、m：月线、y：年线

    // 构造函数
    public ZhituStockKLine() {
    }

    public String getTimeLevel() {
        return timeLevel;
    }

    public void setTimeLevel(String timeLevel) {
        this.timeLevel = timeLevel;
    }

    // Getters and Setters
    public String getTradeTime() {
        return tradeTime;
    }

    public void setTradeTime(String tradeTime) {
        this.tradeTime = tradeTime;
    }

    public double getOpen() {
        return open;
    }

    public void setOpen(double open) {
        this.open = open;
    }

    public double getHigh() {
        return high;
    }

    public void setHigh(double high) {
        this.high = high;
    }

    public double getLow() {
        return low;
    }

    public void setLow(double low) {
        this.low = low;
    }

    public double getClose() {
        return close;
    }

    public void setClose(double close) {
        this.close = close;
    }

    public double getVolume() {
        return volume;
    }

    public void setVolume(double volume) {
        this.volume = volume;
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public double getPrevClose() {
        return prevClose;
    }

    public void setPrevClose(double prevClose) {
        this.prevClose = prevClose;
    }

    public String getIsSuspended() {
        return isSuspended;
    }

    public void setIsSuspended(String isSuspended) {
        this.isSuspended = isSuspended;
    }

    public String getStockCode() {
        return stockCode;
    }

    public void setStockCode(String stockCode) {
        this.stockCode = stockCode;
    }

    @Override
    public String toString() {
        return "ZhituStockKLine{" +
                "stockCode='" + stockCode + '\'' +
                ", tradeTime='" + tradeTime + '\'' +
                ", open=" + open +
                ", high=" + high +
                ", low=" + low +
                ", close=" + close +
                ", volume=" + volume +
                ", amount=" + amount +
                ", prevClose=" + prevClose +
                ", isSuspended='" + isSuspended + '\'' +
                ", timeLevel='" + timeLevel + '\'' +
                '}';
    }
}
