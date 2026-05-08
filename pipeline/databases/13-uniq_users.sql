-- 13-uniq_users.sql
-- Creates a table users with
-- id (auto increment, primary key),
-- email (unique, not null),
-- and name

CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    PRIMARY KEY (id)
);
