package com.sunyuyang.util;

import com.sunyuyang.dao.DatabaseConfig;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class ConfigReaderUtil {
    public static void main(String[] args) {
        ConfigReaderUtil cr = new ConfigReaderUtil();
        System.out.println(cr.getValue("jdbcUrl"));
        System.out.println(cr.getValue("DB_Url"));
        System.out.println(cr.getValue("DB_User"));
        System.out.println(cr.getValue("DB_Pass"));
    }

    public String getValue(String keyName) {
        Properties properties = new Properties();
        InputStream inputStream = null;
        String result;
        try {
            // 加载配置文件
            inputStream = DatabaseConfig.class.getClassLoader()
                    .getResourceAsStream("config.properties");
            if (inputStream == null) {
                throw new RuntimeException("找不到 config.properties 配置文件");
            }
            properties.load(inputStream);
            result = properties.getProperty(keyName);
        } catch (IOException e) {
            throw new RuntimeException("加载配置文件异常", e);
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    System.out.println("关闭配置文件异常：" + e);
                }
            }
        }
        return result;
    }
}
