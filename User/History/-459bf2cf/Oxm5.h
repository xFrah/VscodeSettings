// uart_events_example.h
#ifndef UART_EVENTS_EXAMPLE_H
#define UART_EVENTS_EXAMPLE_H

#include "driver/uart.h"
#include "esp_timer.h"
#include "freertos/queue.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EXPIRY_DATE 0
#define PRODUCT_ID 1
#define MAX_PAYLOAD_SIZE 700
#define MIN_PACKET_LENGTH 5
#define TYPE_COMMAND 0x00
#define TYPE_RESPONSE 0x01
#define TYPE_NOTICE 0x02
#define COMMAND_NOTICE 0x22
#define COMMAND_ERROR 0xFF
#define COMMAND_GET_TX_POWER 0xB7
#define COMMAND_SET_TX_POWER 0xB6
#define COMMAND_GET_REGION 0x08
#define COMMAND_GET_MODEM_PARAMETERS 0xF1
#define COMMAND_SET_MODEM_PARAMETERS 0xF0
#define COMMAND_SET_FREQUENCY_HOPPING 0xAD

#define EX_UART_NUM UART_NUM_0
#define BUF_SIZE (1024)
#define RD_BUF_SIZE (BUF_SIZE)

struct tag {
    char id[12];
    int timestamp;
};

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
