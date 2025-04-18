args <- commandArgs(trailingOnly = TRUE)
geojson_path <- args[1]
output_path <- args[2]

library(sf)
library(terra)
library(dplyr)
library(nngeo)

# Load individual raster layers separately
ALSTK <- rast("/src/planning/species_info/ALSTK.tiff")
FLDTYPCD <- rast("/src/planning/species_info/FLDTYPCD.tiff")

# Load polygon
poly <- st_read(geojson_path) %>% st_transform(crs(ALSTK))

# Extract values at all cells covered by the polygon
extracted <- extract(c(ALSTK, FLDTYPCD), vect(poly), cells = TRUE)

# Get coordinates of the cells
coords <- xyFromCell(ALSTK, extracted$cell)

# Create data frame with cell info, coordinates, and values
cells_df <- data.frame(
  cell = extracted$cell,
  x = coords[,1],
  y = coords[,2],
  ALSTK = extracted[,2],
  FLDTYPCD = extracted[,3]
)

# Replace null (NA) ALSTK with 0
cells_df$ALSTK[cells_df$ALSTK == -32768] <- 0
cells_df$FLDTYPCD[cells_df$FLDTYPCD == -32768] <- NA

# Determine if regeneration treatment is needed
cells_df <- cells_df %>% mutate(
  treatment = if_else(ALSTK < 60, TRUE, FALSE)
)

# Function to fill nearest non-null FLDTYPCD if missing
fill_nearest_fldtypcd <- function(df) {
  na_cells <- which(is.na(df$FLDTYPCD))
  valid_cells <- which(!is.na(df$FLDTYPCD))
  
  if(length(na_cells) > 0 && length(valid_cells) > 0){
    coords_na <- df[na_cells, c("x", "y")]
    coords_valid <- df[valid_cells, c("x", "y")]
    
    nearest_idx <- st_nn(st_as_sf(coords_na, coords = c("x", "y")), 
                         st_as_sf(coords_valid, coords = c("x", "y")), 
                         k = 1, returnDist = FALSE)
    
    for(i in seq_along(na_cells)){
      df$FLDTYPCD[na_cells[i]] <- df$FLDTYPCD[valid_cells[nearest_idx[[i]][1]]]
    }
  }
  return(df)
}

cells_df <- fill_nearest_fldtypcd(cells_df)

# Lookup tables
fvs_species_lookup <- list(
  `103` = c("easternwhitepine", "redmaple", "paperbirch", "sweetbirch", "yellowbirch", "blackcherry", "whiteash", "northernredoak", "sugarmaple", "basswood", "easternhemlock", "northernwhitecedar", "yellowpoplar", "whiteoak", "chestnutoak", "scarletoak", "shortleafpine"),
  `402` = c("easternredcedar", "whiteoak", "hickoryspp", "blackwalnut", "whiteash", "blacklocust", "floweringdogwood", "blackgum", "hackberry", "shortleafpine"),
  `503` = c("whiteoak", "pinoak", "chinkapinoak", "buroak", "northernredoak", "sugarmaple", "redmaple", "blackwalnut", "basswood", "blacklocust", "sweetgum", "blackgum", "yellowpoplar", "floweringdogwood"),
  `504` = c("whiteoak", "buroak", "northernredoak", "hickoryspp", "whiteash", "yellowpoplar"),
  `505` = c("northernredoak", "buroak", "scarletoak", "chestnutoak", "yellowpoplar"),
  `506` = c("yellowpoplar", "whiteoak", "northernredoak", "buroak", "easternhemlock", "blackgum", "hickoryspp"),
  `509` = c("buroak", "pinoak", "chinkapinoak", "easternredcedar", "shagbarkhickory", "blackwalnut", "easterncottonwood", "whiteash", "americanelm", "swampwhiteoak", "blacklocust", "basswood"),
  `511` = c("yellowpoplar", "blacklocust", "redmaple", "sweetbirch", "cucumbertree", "whiteoak", "northernredoak"),
  `512` = c("blackwalnut", "yellowpoplar", "whiteash", "blackcherry", "basswood", "sugarmaple", "whiteoak", "hickoryspp"),
  `513` = c("blacklocust", "whiteoak", "redmaple", "hickoryspp", "sweetbirch", "shortleafpine"),
  `802` = c("blackcherry", "sugarmaple", "northernredoak", "redmaple", "basswood", "sweetbirch", "americanelm", "easternhemlock"),
  `519` = c("redmaple", "whiteoak", "northernredoak", "hickoryspp", "yellowpoplar", "blacklocust", "sassafras", "VP", "shortleafpine"),
  `520` = c("floweringdogwood", "hackberry", "blacklocust", "blackgum", "sourwood", "southernredoak", "shingleoak", "wateroak", "blackwalnut"),
  `708` = c("redmaple", "yellowpoplar", "blackgum", "sweetgum", "loblollypine"),
  `801` = c("sugarmaple", "BE", "yellowbirch", "basswood", "redmaple", "easternhemlock", "northernredoak", "whiteash", "WP", "blackcherry", "sweetbirch", "americanelm", "easternhophornbeam"),
  `809` = c("redmaple", "sugarmaple", "BE", "paperbirch", "easternwhitepine", "redpine", "easternhemlock")
)

crown_diameter_lookup <- list(
  easternwhitepine = 13.7, redpine = 9.1, redmaple = 16.2, paperbirch = 13.1,
  sweetbirch = 16.8, yellowbirch = 11.9, blackcherry = 15.8, whiteash = 19.2,
  northernredoak = 25.0, sugarmaple = 16.8, basswood = 18.6, easternhemlock = 12.2,
  northernwhitecedar = 8.2, yellowpoplar = 18.6, whiteoak = 21.0, chestnutoak = 20.8,
  scarletoak = 20.4, shortleafpine = 10.7, easternredcedar = 10.4, hickoryspp = 16.8,
  blackwalnut = 11.6, blacklocust = 14.6, floweringdogwood = 11.3, blackgum = 15.5,
  hackberry = 15.5, pinoak = 13.4, chinkapinoak = 14.0, buroak = 18.9, shagbarkhickory = 16.8,
  easterncottonwood = 25.9, americanelm = 30.2, swampwhiteoak = 21.3, sweetgum = 15.2,
  cucumbertree = 11.9, loblollypine = 17.1, shingleoak = 16.8, wateroak = 17.7,
  southernredoak = 17.4, sourwood = 11.3, easternhophornbeam = 11.9, sassafras = 8.8
)

# Assign species
assign_fvs_species <- function(fldtypcd) {
  fldtypcd_chr <- as.character(fldtypcd)
  if (!is.na(fldtypcd_chr) && fldtypcd_chr %in% names(fvs_species_lookup)) {
    sample(fvs_species_lookup[[fldtypcd_chr]], 1)
  } else {
    NA
  }
}

# Assign crown diameter
get_crown_diameter <- function(species_code) {
  if (!is.na(species_code) && species_code %in% names(crown_diameter_lookup)) {
    return(crown_diameter_lookup[[species_code]])
  } else {
    return(NA)
  }
}

cells_df$FVS_Species <- sapply(cells_df$FLDTYPCD, assign_fvs_species)
cells_df$Crown_Diameter_m <- sapply(cells_df$FVS_Species, get_crown_diameter)

# Calculate number of trees per 30x30m cell (900 m²)
# Each tree requires π * (crown_diameter / 2)^2 space
cells_df <- cells_df %>%
  mutate(
    tree_area = pi * (Crown_Diameter_m / 2)^2,
    trees_to_plant = ifelse(treatment, floor(900 / tree_area), 0)
  )

# Summarize results by species
species_summary <- cells_df %>%
  filter(treatment == TRUE, !is.na(FVS_Species)) %>%
  group_by(FVS_Species) %>%
  summarize(
    Total_Trees_to_Plant = sum(trees_to_plant, na.rm = TRUE),
    Crown_Diameter_m = max(Crown_Diameter_m, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(species_summary, output_path, row.names = FALSE)