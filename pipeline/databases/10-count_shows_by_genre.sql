-- 10-count_shows_by_genre.sql
-- Lists all genres and the number of shows linked to each
-- Only displays genres that have at least one show linked
-- Results sorted in descending order by the number of shows
SELECT tv_genres.name AS genre, COUNT(tv_show_genres.show_id) AS number_of_shows
FROM tv_genres
INNER JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
GROUP BY tv_genres.name
ORDER BY number_of_shows DESC;
