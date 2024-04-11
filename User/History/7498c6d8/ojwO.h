// rfid_events.c or nfc_events.c
#include "events.h"

// Define the event base for RFID events
ESP_EVENT_DEFINE_BASE(RFID_EVENT);
// Define the event base for NFC events
ESP_EVENT_DEFINE_BASE(NFC_EVENT);