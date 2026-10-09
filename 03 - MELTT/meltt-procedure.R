#####################################################################
# MELTT pipeline for merging cleaned ACLED and UCDP data
#####################################################################

suppressPackageStartupMessages({
  requireNamespace("meltt", quietly = TRUE)
  requireNamespace("readxl", quietly = TRUE)
  requireNamespace("dplyr", quietly = TRUE)
})

library(dplyr)

project_root <- function() {
  env_root <- Sys.getenv("PROJECT_ROOT")
  if (nzchar(env_root)) {
    return(normalizePath(env_root, winslash = "/", mustWork = TRUE))
  }

  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

project_dir <- project_root()

data_dir <- file.path(project_dir, "01 - Data", "02 - Source_Data-cleaned")
meltt_dir <- file.path(project_dir, "03 - MELTT")

acled_path <- file.path(data_dir, "acled_cleaned.csv")
ucdp_path <- file.path(data_dir, "UCDP_cleaned_russia_involvement.csv")
violence_path <- file.path(meltt_dir, "violence_type.xlsx")
prec_path <- file.path(meltt_dir, "prec_tax.csv")

stopifnot(file.exists(acled_path), file.exists(ucdp_path), file.exists(violence_path), file.exists(prec_path))

ucdp <- read.csv(ucdp_path, stringsAsFactors = FALSE)
acled <- read.csv(acled_path, stringsAsFactors = FALSE)
violence_tax <- readxl::read_excel(violence_path)
prec_tax <- read.csv(prec_path, sep = ";", stringsAsFactors = FALSE)

taxonomy <- list(prec_tax = prec_tax, violence_tax = violence_tax)

if (!"violence_tax" %in% names(acled) && "interaction" %in% names(acled)) {
  acled$violence_tax <- acled$interaction
}
if (!"where_prec" %in% names(acled) && "geo_precision" %in% names(acled)) {
  acled$where_prec <- acled$geo_precision
}
if (!"date_prec" %in% names(acled) && "time_precision" %in% names(acled)) {
  acled$date_prec <- acled$time_precision
}

acled <- acled %>%
  mutate(
    date = as.Date(date_start),
    enddate = as.Date(date_start),
    violence_tax = as.character(violence_tax),
    prec_tax = as.character(if ("where_prec" %in% names(.)) where_prec else if ("geo_precision" %in% names(.)) geo_precision else NA_character_),
    date_prec = as.character(if ("date_prec" %in% names(.)) date_prec else if ("time_precision" %in% names(.)) time_precision else NA_character_)
  ) %>%
  as.data.frame()

ucdp <- ucdp %>%
  mutate(
    date = as.Date(date_start),
    enddate = as.Date(date_end),
    violence_tax = as.character(type_of_violence),
    prec_tax = as.character(where_prec)
  ) %>%
  as.data.frame()

# meltt() resolves bare dataset names from the calling environment.
.GlobalEnv$acled <- acled
.GlobalEnv$ucdp <- ucdp

output <- meltt::meltt(
  acled,
  ucdp,
  taxonomies = taxonomy,
  twindow = 1,
  spatwindow = 3,
  silent = TRUE
)

duplicates <- tryCatch(
  meltt::meltt_duplicates(output),
  error = function(e) {
    message("meltt_duplicates() failed in this package version; continuing without duplicate export.")
    data.frame()
  }
)

dataset <- meltt::meltt_data(output, return_all = TRUE)
plot(output)

write.csv(duplicates, file.path(data_dir, "meltt_duplicates.csv"), row.names = FALSE)
write.csv(dataset, file.path(data_dir, "dataset_meltt.csv"), row.names = FALSE)

tab <- with(dataset, table(dataset, year))
write.csv(tab, file.path(data_dir, "table_2026.csv"))

message("MELTT pipeline completed successfully.")
message("Duplicates saved to: ", file.path(data_dir, "meltt_duplicates.csv"))
message("Dataset saved to: ", file.path(data_dir, "dataset_meltt.csv"))

