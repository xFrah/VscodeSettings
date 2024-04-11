#include "fdb_def.h"
// fdb_tsl_set_status
/**
 * Set the TSL status.
 *
 * @param db database object
 * @param tsl TSL object
 * @param status status
 *
 * @return result
 */
fdb_err_t fdb_tsl_set_status(fdb_tsdb_t db, fdb_tsl_t tsl, fdb_tsl_status_t status);
