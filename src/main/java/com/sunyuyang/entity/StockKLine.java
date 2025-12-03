package com.sunyuyang.entity;

import java.time.LocalDateTime;

public class StockKLine {
    private LocalDateTime tradeTime;
    private double open;
    private double high;
    private double low;
    private double close;
    private double volume;
    private double amount;
    private double prevClose;
    private boolean isSuspended;
    private String stockCode;

    // 构造函数
    public StockKLine() {
    }

    public StockKLine(LocalDateTime tradeTime, double open, double high, double low,
                      double close, double volume, double amount,
                      double prevClose, boolean isSuspended, String stockCode) {
        this.tradeTime = tradeTime;
        this.open = open;
        this.high = high;
        this.low = low;
        this.close = close;
        this.volume = volume;
        this.amount = amount;
        this.prevClose = prevClose;
        this.isSuspended = isSuspended;
        this.stockCode = stockCode;
    }

    // Getters and Setters
    public LocalDateTime getTradeTime() {
        return tradeTime;
    }

    public void setTradeTime(LocalDateTime tradeTime) {
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

    public boolean isSuspended() {
        return isSuspended;
    }

    public void setSuspended(boolean suspended) {
        isSuspended = suspended;
    }

    public String getStockCode() {
        return stockCode;
    }

    public void setStockCode(String stockCode) {
        this.stockCode = stockCode;
    }

    @Override
    public String toString() {
        return String.format("StockKLine{time=%s, close=%.4f, volume=%.0f}",
                tradeTime, close, volume);
    }
}
