-- 9-no_genre.sql
-- Lists all shows that have no genre linked,
-- displaying title and genre_id (NULL)
-- Sorted in ascending order by title and genre_id
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
WHERE tv_show_genres.genre_id IS NULL
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
