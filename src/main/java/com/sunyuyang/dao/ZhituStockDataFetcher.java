package com.sunyuyang.dao;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.sunyuyang.entity.ZhituStockKLine;
import com.sunyuyang.util.ConfigReaderUtil;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

public class ZhituStockDataFetcher {
    private static final OkHttpClient OK_HTTP_CLIENT = new OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS) // 智图接口响应稍慢，适当延长超时
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(15, TimeUnit.SECONDS)
            .build();
    ConfigReaderUtil cr = new ConfigReaderUtil();

    /**
     * 测试方法：直接运行获取数据
     */
    public static void main(String[] args) {
        ZhituStockDataFetcher fetcher = new ZhituStockDataFetcher();
        ZhituStockDataDao dataDao = new ZhituStockDataDao();
        try {
            // 示例：获取 贵州茅台（600519，沪市）2025年11月1日-2025年11月18日的 5分钟前复权数据
            List<ZhituStockKLine> stockTimeList = fetcher.getStockTimeList("300624.SZ", "5", "n", "20240101", "20250101");
            dataDao.saveKLineDataBatch(stockTimeList);
        } catch (IOException e) {
            System.err.println("数据获取失败（网络/接口连接问题）：" + e.getMessage());
        } catch (RuntimeException e) {
            System.err.println("数据解析失败（接口响应/参数问题）：" + e.getMessage());
        }
    }

    /**
     * 核心方法：获取基础的股票代码和名称，用于后续接口的参数传入。
     *
     * @return 基础的股票代码和名称
     * @throws IOException HTTP 请求异常
     */
    public List<ZhituStockKLine> getStockTimeList(String stockCode, String timeLevel, String exRightsType, String startTime, String endTime) throws IOException {
        List<ZhituStockKLine> stockList = new ArrayList<ZhituStockKLine>();
        // 构建请求参数（智图接口要求的 Query 参数格式，非 JSON 体）
        HttpUrl.Builder urlBuilder = Objects.requireNonNull(HttpUrl.parse(cr.getValue("ZHITU_URL_HISTORY_FSJY") + stockCode + "/" + timeLevel + "/" + exRightsType)).newBuilder();
        urlBuilder.addQueryParameter("token", cr.getValue("ZHITU_TOKEN"));
        urlBuilder.addQueryParameter("st", startTime);
        urlBuilder.addQueryParameter("et", endTime);
        // 构建 GET 请求（智图接口默认支持 GET，部分版本支持 POST，需按官方文档调整）
        Request request = new Request.Builder()
                .url(urlBuilder.build())
                .get()
                .addHeader("Content-Type", "application/json")
                .build();
        // 执行请求并解析响应
        try (Response response = OK_HTTP_CLIENT.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("HTTP 请求失败：状态码 " + response.code() + "，原因：" + response.message());
            }

            // 读取响应体并解析 JSON
            String responseBody = Objects.requireNonNull(response.body()).string();
            JSONArray resultJson = JSON.parseArray(responseBody);
            System.out.println("resultJson.size:" + resultJson.size());
            // 遍历数组元素
            for (int i = 0; i < resultJson.size(); i++) {
                JSONObject obj = resultJson.getJSONObject(i);
                ZhituStockKLine stockKLine = new ZhituStockKLine();
                stockKLine.setStockCode(stockCode);//股票代码
                stockKLine.setTradeTime(obj.getString("t")); //交易时间
                stockKLine.setOpen(obj.getDouble("o"));//开盘价
                stockKLine.setHigh(obj.getDouble("h"));//最高价
                stockKLine.setLow(obj.getDouble("l"));//最低价
                stockKLine.setClose(obj.getDouble("c"));//收盘价
                stockKLine.setVolume(obj.getDouble("v"));//成交量
                stockKLine.setAmount(obj.getDouble("a"));//成交额
                stockKLine.setPrevClose(obj.getDouble("pc"));//前收盘价
                stockKLine.setIsSuspended(obj.getString("sf"));//停牌 1停牌，0 不停牌
                stockKLine.setTimeLevel(timeLevel);
                //System.out.println(stockKLine.toString());
                stockList.add(stockKLine);
            }
        }
        return stockList;
    }
}
