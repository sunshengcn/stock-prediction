package com.sunyuyang.dao;

import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;

import javax.sql.DataSource;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class DatabaseConfig {
    private static HikariDataSource dataSource;

    static {
        try {
            Properties props = new Properties();
            InputStream input = DatabaseConfig.class.getClassLoader()
                    .getResourceAsStream("database.properties");

            if (input == null) {
                // 使用默认配置
                props.setProperty("jdbcUrl", "jdbc:mysql://localhost:3306/stock_prediction?useSSL=false&serverTimezone=UTC");
                props.setProperty("username", "root");
                props.setProperty("password", "password");
                props.setProperty("maximumPoolSize", "10");
                props.setProperty("minimumIdle", "5");
                props.setProperty("connectionTimeout", "30000");
                props.setProperty("idleTimeout", "600000");
                props.setProperty("maxLifetime", "1800000");
            } else {
                props.load(input);
            }

            HikariConfig config = new HikariConfig();
            config.setJdbcUrl(props.getProperty("jdbcUrl"));
            config.setUsername(props.getProperty("username"));
            config.setPassword(props.getProperty("password"));
            config.setMaximumPoolSize(Integer.parseInt(props.getProperty("maximumPoolSize", "10")));
            config.setMinimumIdle(Integer.parseInt(props.getProperty("minimumIdle", "5")));
            config.setConnectionTimeout(Long.parseLong(props.getProperty("connectionTimeout", "30000")));
            config.setIdleTimeout(Long.parseLong(props.getProperty("idleTimeout", "600000")));
            config.setMaxLifetime(Long.parseLong(props.getProperty("maxLifetime", "1800000")));

            dataSource = new HikariDataSource(config);

        } catch (IOException e) {
            throw new RuntimeException("Failed to load database configuration", e);
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
}
