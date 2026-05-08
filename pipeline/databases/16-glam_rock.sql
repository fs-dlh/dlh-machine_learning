-- 16-glam_rock.sql
-- Lists all bands with Glam rock as main style, ordered by decreasing lifespan

SELECT
    band_name,
    (COALESCE(split, 2020) - formed) AS lifespan
FROM
    metal_bands
WHERE
    style LIKE '%Glam rock%'
ORDER BY
    lifespan DESC;
    