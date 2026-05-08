-- 14-country_users.sql
-- Creates a table users with
--     id (auto increment, primary key),
--     email (unique, not null),
--     name,
--     and country enumeration (US, CO, TN)
-- The country column has a default value of 'US' and cannot be NULL
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US',
    PRIMARY KEY (id)
);