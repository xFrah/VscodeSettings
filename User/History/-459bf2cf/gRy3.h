// uart_events_example.h
#ifndef UART_EVENTS_EXAMPLE_H
#define UART_EVENTS_EXAMPLE_H

#include "driver/uart.h"
#include "esp_timer.h"
#include "freertos/queue.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Function prototypes
int tag_compare(const void *a, const void *b, void *udata);
uint64_t tag_hash(const void *item, uint64_t seed0, uint64_t seed1);
const char *getRegionName(uint8_t regionCode);
const char *getResponseCode(uint8_t responseCode);
int search_for_bytes(uint8_t *buffer, int buffer_size, uint8_t *pattern, int pattern_size, int offset);
bool validate_packet(const uint8_t *packet, int length);
void decode(uint8_t *raw);
void reset_reader_variables(int *data_buffer_size, int *packet_size);
void uart_event_task(void *pvParameters);
void scan_tags();
struct hashmap *rfid_init(void);

#ifdef __cplusplus
}
#endif

#endif // UART_EVENTS_EXAMPLE_H
