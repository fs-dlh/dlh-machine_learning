-- 12-rating_genres.sql
-- Lists all genres by their total rating (sum of ratings)
-- Results sorted in descending order by rating
SELECT tv_genres.name, COALESCE(SUM(tv_show_ratings.rate), 0) AS rating
FROM tv_genres
INNER JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
INNER JOIN tv_show_ratings ON tv_show_genres.show_id = tv_show_ratings.show_id
GROUP BY tv_genres.id
ORDER BY rating DESC;
