#ifndef NFC_EVENT_DEFS_H
#define NFC_EVENT_DEFS_H

#include "esp_event.h"

ESP_EVENT_DECLARE_BASE(NFC_EVENT);  // Declare the base for NFC events

// NFC Events
typedef enum {
    NFC_EVENT_TAG_FOUND, // Indicates an NFC tag has been found
} nfc_event_t;

#endif // NFC_EVENT_DEFS_H
