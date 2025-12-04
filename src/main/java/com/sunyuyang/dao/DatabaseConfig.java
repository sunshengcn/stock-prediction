package com.sunyuyang.dao;

import com.sunyuyang.util.ConfigReaderUtil;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;

import javax.sql.DataSource;

public class DatabaseConfig {
    private static final HikariDataSource dataSource;

    static {
        try {
            ConfigReaderUtil cr = new ConfigReaderUtil();
            HikariConfig config = new HikariConfig();
            config.setJdbcUrl(cr.getValue("DB_Url"));
            config.setUsername(cr.getValue("DB_User"));
            config.setPassword(cr.getValue("DB_Pass"));
            config.setMaximumPoolSize(Integer.parseInt(cr.getValue("maximumPoolSize")));
            config.setMinimumIdle(Integer.parseInt(cr.getValue("minimumIdle")));
            config.setConnectionTimeout(Long.parseLong(cr.getValue("connectionTimeout")));
            config.setIdleTimeout(Long.parseLong(cr.getValue("idleTimeout")));
            config.setMaxLifetime(Long.parseLong(cr.getValue("maxLifetime")));

            dataSource = new HikariDataSource(config);

        } catch (Exception e) {
            throw new RuntimeException("读取配置文件异常", e);
        }
    }

    public static DataSource getDataSource() {
        return dataSource;
    }

    public static void closeDataSource() {
        if (dataSource != null && !dataSource.isClosed()) {
            dataSource.close();
        }
    }

    public static void main(String[] args) {
        System.out.println("dataSource:" + dataSource);
    }
}
