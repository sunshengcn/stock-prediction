package com.sunyuyang.dao;

import com.sunyuyang.entity.ZhituStockKLine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class ZhituStockDataDao {
    private static final Logger logger = LoggerFactory.getLogger(ZhituStockDataDao.class);
    private final DataSource dataSource;

    public ZhituStockDataDao() {
        this.dataSource = DatabaseConfig.getDataSource();
    }

    public ZhituStockDataDao(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    /**
     * 获取指定股票和时间范围的K线数据
     */
    public List<ZhituStockKLine> getKLineData(String stockCode, LocalDateTime startDate, LocalDateTime endDate) {
        String sql = "SELECT trade_time, open_price, high_price, low_price, close_price, " +
                "volume, amount, prev_close, is_suspended,time_level " +
                "FROM zhitu_stock_k_line " +
                "WHERE stock_code = ? AND trade_time BETWEEN ? AND ? " +
                "ORDER BY trade_time ASC";

        List<ZhituStockKLine> result = new ArrayList<>();

        try (Connection conn = dataSource.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, stockCode);
            ps.setTimestamp(2, Timestamp.valueOf(startDate));
            ps.setTimestamp(3, Timestamp.valueOf(endDate));

            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    ZhituStockKLine kline = new ZhituStockKLine();
                    kline.setStockCode(stockCode);
                    kline.setTradeTime(rs.getString("trade_time"));
                    kline.setOpen(rs.getDouble("open_price"));
                    kline.setHigh(rs.getDouble("high_price"));
                    kline.setLow(rs.getDouble("low_price"));
                    kline.setClose(rs.getDouble("close_price"));
                    kline.setVolume(rs.getDouble("volume"));
                    kline.setAmount(rs.getDouble("amount"));
                    kline.setPrevClose(rs.getDouble("prev_close"));
                    kline.setIsSuspended(rs.getString("is_suspended"));
                    kline.setTimeLevel(rs.getString("time_level"));
                    result.add(kline);
                }
            }

            logger.info("Fetched {} records for stock {} from {} to {}",
                    result.size(), stockCode, startDate, endDate);

        } catch (SQLException e) {
            logger.error("Failed to fetch K-line data for stock {}", stockCode, e);
            throw new RuntimeException("Failed to fetch K-line data", e);
        }

        return result;
    }

    /**
     * 批量保存K线数据
     */
    public void saveKLineDataBatch(List<ZhituStockKLine> klineData) {
        if (klineData == null || klineData.isEmpty()) {
            return;
        }

        String sql = "INSERT INTO zhitu_stock_k_line " +
                "(stock_code, trade_time, open_price, high_price, low_price, " +
                "close_price, volume, amount, prev_close, is_suspended,time_level) " +
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?) " +
                "ON DUPLICATE KEY UPDATE " +
                "open_price = VALUES(open_price), high_price = VALUES(high_price), " +
                "low_price = VALUES(low_price), close_price = VALUES(close_price), " +
                "volume = VALUES(volume), amount = VALUES(amount), " +
                "prev_close = VALUES(prev_close), is_suspended = VALUES(is_suspended)," +
                "time_level = VALUES(time_level)";

        try (Connection conn = dataSource.getConnection();
        PreparedStatement ps = conn.prepareStatement(sql)) {
            conn.setAutoCommit(false);

            for (ZhituStockKLine kline : klineData) {
                ps.setString(1, kline.getStockCode());
                ps.setTimestamp(2, Timestamp.valueOf(kline.getTradeTime()));
                ps.setDouble(3, kline.getOpen());
                ps.setDouble(4, kline.getHigh());
                ps.setDouble(5, kline.getLow());
                ps.setDouble(6, kline.getClose());
                ps.setDouble(7, kline.getVolume());
                ps.setDouble(8, kline.getAmount());
                ps.setDouble(9, kline.getPrevClose());
                ps.setString(10, kline.getIsSuspended());
                ps.setString(11, kline.getTimeLevel());
                ps.addBatch();
            }

            int[] results = ps.executeBatch();
            conn.commit();

            logger.info("Successfully saved {} records to database", results.length);

        } catch (SQLException e) {
            logger.error("Failed to save K-line data batch", e);
            throw new RuntimeException("Failed to save K-line data", e);
        }
    }

    /**
     * 获取最新交易时间
     */
    public LocalDateTime getLatestTradeTime(String stockCode) {
        String sql = "SELECT MAX(trade_time) as latest_time FROM zhitu_stock_k_line WHERE stock_code = ?";

        try (Connection conn = dataSource.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, stockCode);

            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next() && rs.getTimestamp("latest_time") != null) {
                    return rs.getTimestamp("latest_time").toLocalDateTime();
                }
            }

        } catch (SQLException e) {
            logger.error("Failed to get latest trade time for stock {}", stockCode, e);
        }

        return LocalDateTime.now().minusYears(3); // 默认返回3年前
    }

    /**
     * 检查表是否存在
     */
    public boolean checkTableExists() {
        String sql = "SELECT COUNT(*) FROM information_schema.tables " +
                "WHERE table_schema = DATABASE() AND table_name = 'zhitu_stock_k_line'";

        try (Connection conn = dataSource.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {

            if (rs.next()) {
                return rs.getInt(1) > 0;
            }

        } catch (SQLException e) {
            logger.error("Failed to check table existence", e);
        }

        return false;
    }
}
