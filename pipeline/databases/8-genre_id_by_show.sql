-- 8-genre_id_by_show.sql
-- Lists all shows that have at least one genre linked,
-- displaying title and genre_id
-- Sorted in ascending order by title and genre_id
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
INNER JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
