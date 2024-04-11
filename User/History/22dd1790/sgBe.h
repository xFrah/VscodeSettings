#ifndef RFID_EVENT_DEFS_H
#define RFID_EVENT_DEFS_H

#include "esp_event.h"

ESP_EVENT_DECLARE_BASE(RFID_EVENT); // Declare the base for RFID events

// RFID Events
typedef enum {
    RFID_EVENT_SCAN_FINISHED, // Indicates RFID scan is finished
} rfid_event_t;

#endif // RFID_EVENT_DEFS_H
