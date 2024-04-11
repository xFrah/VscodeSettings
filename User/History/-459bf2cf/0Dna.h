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

struct tag {
    char id[13];
    int timestamp;
};

void scan_tags();
struct hashmap *rfid_init(void);

#ifdef __cplusplus
}
#endif

#endif // UART_EVENTS_EXAMPLE_H
