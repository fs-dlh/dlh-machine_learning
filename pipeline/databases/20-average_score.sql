-- 20-average_score.sql: Compute and store average score for a student
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id INT)
BEGIN
    DECLARE avg_score FLOAT;

    -- Calculate average score from corrections table
    SELECT AVG(score) INTO avg_score
    FROM corrections
    WHERE corrections.user_id = user_id;

    -- If no corrections exist, set average to 0
    IF avg_score IS NULL THEN
        SET avg_score = 0;
    END IF;

    -- Update the user's average_score field
    UPDATE users
    SET average_score = avg_score
    WHERE id = user_id;
END //

DELIMITER ;
