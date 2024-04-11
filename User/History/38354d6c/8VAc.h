#ifndef EVENT_DEFS_H
#define EVENT_DEFS_H

#include "esp_event.h"

ESP_EVENT_DECLARE_BASE(RFID_EVENT); // Declare the base for RFID events
ESP_EVENT_DECLARE_BASE(NFC_EVENT);  // Declare the base for NFC events

// RFID Events
typedef enum {
    RFID_EVENT_SCAN_FINISHED, // Indicates RFID scan is finished
} rfid_event_t;

// NFC Events
typedef enum {
    NFC_EVENT_TAG_FOUND, // Indicates an NFC tag has been found
} nfc_event_t;

#endif // EVENT_DEFS_H
