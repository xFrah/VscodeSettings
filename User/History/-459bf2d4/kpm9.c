#include "m100.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "hashmap.h"
#include "rfid_events.h"
#include <esp_err.h>
#include <stdio.h>
#include <string.h>

static const char *TAG = "rfid";

#define EXPIRY_DATE 0
#define PRODUCT_ID 1
#define MAX_PAYLOAD_SIZE 128
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

#define RFID_UART_NUM UART_NUM_1
#define EX_UART_NUM UART_NUM_1

#define BUF_SIZE (1024)
#define RD_BUF_SIZE (BUF_SIZE)

static uint8_t data_buffer[RD_BUF_SIZE * 2];
static int data_buffer_size = 0;

static QueueHandle_t uart0_queue;

struct hashmap *tag_map;

void uart_write_bytes_with_delay(const uint8_t *data, size_t length, uint32_t wait_ms) {
    vTaskDelay(pdMS_TO_TICKS(wait_ms));
    uart_write_bytes(RFID_UART_NUM, (const char *)data, length);
}

void set_frequency_hopping(uint32_t wait_ms) {
    const uint8_t cmd[] = {0xBB, 0x00, 0xAD, 0x00, 0x01, 0xFF, 0xAD, 0x7E};
    uart_write_bytes_with_delay(cmd, sizeof(cmd), wait_ms);
}

void get_tx_power(uint32_t wait_ms) {
    const uint8_t cmd[] = {0xBB, 0x00, 0xB7, 0x00, 0x00, 0xB7, 0x7E};
    uart_write_bytes_with_delay(cmd, sizeof(cmd), wait_ms);
}

void set_tx_power(int power, uint32_t wait_ms) {
    uint8_t msb = power >> 8;
    uint8_t lsb = power & 0xFF;
    uint8_t checksum = 0x00 + 0xB6 + 0x00 + 0x02 + msb + lsb;
    checksum &= 0xFF;
    const uint8_t cmd[] = {0xBB, 0x00, 0xB6, 0x00, 0x02, msb, lsb, checksum, 0x7E};
    uart_write_bytes_with_delay(cmd, sizeof(cmd), wait_ms);
}

void set_modem_params(uint8_t mix_gain, uint8_t if_gain, int threshold, uint32_t wait_ms) {
    uint8_t threshold_msb = threshold >> 8;
    uint8_t threshold_lsb = threshold & 0xFF;
    uint8_t checksum = 0x00 + 0xF0 + 0x00 + 0x04 + mix_gain + if_gain + threshold_msb + threshold_lsb;
    checksum &= 0xFF;
    const uint8_t cmd[] = {0xBB, 0x00, 0xF0, 0x00, 0x04, mix_gain, if_gain, threshold_msb, threshold_lsb, checksum, 0x7E};
    uart_write_bytes_with_delay(cmd, sizeof(cmd), wait_ms);
}

void get_modem_params(uint32_t wait_ms) {
    const uint8_t cmd[] = {0xBB, 0x00, 0xF1, 0x00, 0x00, 0xF1, 0x7E};
    uart_write_bytes_with_delay(cmd, sizeof(cmd), wait_ms);
}

void get_region(uint32_t wait_ms) {
    const uint8_t cmd[] = {0xBB, 0x00, 0x08, 0x00, 0x00, 0x08, 0x7E};
    uart_write_bytes_with_delay(cmd, sizeof(cmd), wait_ms);
}

void start_scan(uint32_t wait_ms) {
    const uint8_t scanCommand[] = {0xBB, 0x00, 0x27, 0x00, 0x03, 0x22, 0x27, 0x10, 0x83, 0x7E};
    uart_write_bytes_with_delay(scanCommand, sizeof(scanCommand), wait_ms); // Convert wait to milliseconds
}

void stop_scan(uint32_t wait_ms) {
    const uint8_t stopScanCommand[] = {0xBB, 0x00, 0x28, 0x00, 0x00, 0x28, 0x7E};
    uart_write_bytes_with_delay(stopScanCommand, sizeof(stopScanCommand), wait_ms); // Convert wait to milliseconds
}

void setup_rfid_module(void) {
    ESP_LOGI(TAG, "Setting up RFID module");
    set_tx_power(2600, 700);
    ESP_LOGI(TAG, "TX power set to 26 dBm");
    get_tx_power(1000);
    ESP_LOGI(TAG, "Getting TX power");
    get_region(1000);
    ESP_LOGI(TAG, "Getting region");
    set_modem_params(0x02, 0x06, 0x00B0, 700);
    ESP_LOGI(TAG, "Setting modem parameters");
    get_modem_params(700);
    ESP_LOGI(TAG, "Getting modem parameters");
    set_frequency_hopping(700);
    ESP_LOGI(TAG, "Setting frequency hopping");
}

int tag_compare(const void *a, const void *b, void *udata) {
    const struct tag *ua = a;
    const struct tag *ub = b;
    return strcmp(ua->id, ub->id);
}

uint64_t tag_hash(const void *item, uint64_t seed0, uint64_t seed1) {
    const struct tag *tag_ = item;
    return hashmap_sip(tag_->id, strlen(tag_->id), seed0, seed1);
}

const char *getRegionName(uint8_t regionCode) {
    switch (regionCode) {
    case 0x01:
        return "China 900MHZ";
    case 0x02:
        return "USA";
    case 0x03:
        return "Europe";
    case 0x04:
        return "China 800MHZ";
    case 0x05:
        return "Korea";
    default:
        return "Unknown region";
    }
}

const char *getResponseCode(uint8_t responseCode) {
    switch (responseCode) {
    case 0x09:
        return "Read fail. No tag response or CRC error";
    case 0x10:
        return "Write fail. No tag response or CRC error";
    case 0x13:
        return "Lock or Kill fail. No tag response or CRC error";
    case 0x15:
        return "Inventory fail. No tag response or CRC error";
    case 0x16:
        return "Access fail. May caused by password error";
    case 0x17:
        return "Invalid command";
    case 0x20:
        return "Frequency hopping time out. All channel are occupied";
    default:
        return "Unknown response code";
    }
}

int search_for_bytes(uint8_t *buffer, int buffer_size, uint8_t *pattern, int pattern_size, int offset) {
    int i;
    for (i = offset; i < buffer_size; i++) {
        if (buffer[i] == pattern[0]) {
            if (memcmp(buffer + i, pattern, pattern_size) == 0) { // TODO is this efficient?
                return i;
            }
        }
    }
    return 0;
}

bool validate_packet(const uint8_t *packet, int length) {
    if (length < 3) { // Ensure there's at least one byte to checksum, plus the two checksum bytes
        printf("Packet too short to validate\n");
        return false;
    }

    if (packet[length - 1] != 0x7E) {
        printf("Packet does not end with 0x7E\n");
        return false;
    }

    uint8_t checksum_byte = packet[length - 2]; // The second-last byte is the checksum
    uint8_t calculated_checksum = 0;
    for (size_t i = 1; i < length - 2; i++) { // Start from 1 to exclude the packet start byte
        calculated_checksum += packet[i];
    }
    calculated_checksum &= 0xFF; // Ensure checksum is within a byte

    bool is_valid = (checksum_byte == calculated_checksum);

    if (!is_valid) {
        printf("Checksum failed: %d != %d\n", checksum_byte, calculated_checksum);
    }
    // printf("Computed checksum: %d\n", calculated_checksum);

    return is_valid;
}

void decode(uint8_t *raw) {
    uint8_t packet_type = raw[1];
    uint8_t command = raw[2];

    if (packet_type == TYPE_NOTICE && command == COMMAND_NOTICE) {
        printf("Tag notice - RSSI: %d, PC: [%02x%02x], EPC: ", (int)(-0xFF + raw[5]), raw[6], raw[7]);
        for (int i = 8; i < 20; ++i) {
            printf("%02x", raw[i]);
        }
        struct tag newTag;
        memcpy(newTag.id, &raw[8], 12);
        newTag.id[12] = '\0';
        newTag.timestamp = esp_timer_get_time();
        hashmap_set(tag_map, &newTag);
        int tag_count = hashmap_count(tag_map);
        ESP_LOGI(TAG, "Tag %s added to map, tag count in map: %d", newTag.id, tag_count);
        printf(", CRC: [%02x%02x]\n", raw[20], raw[21]);
    } else if (packet_type == TYPE_RESPONSE) {
        switch (command) {
        case COMMAND_GET_TX_POWER:
            printf("TX power is %d dBm.\n", (raw[5] << 8) + raw[6]);
            break;
        case COMMAND_SET_TX_POWER:
            printf("TX power has been successfully changed.\n");
            break;
        case COMMAND_GET_REGION:
            printf("Region is %s.\n", getRegionName(raw[5]));
            break;
        case COMMAND_ERROR:
            // if (raw[5] != 0x15) {
            printf("%s\n", getResponseCode(raw[5]));
            // }
            break;
        case COMMAND_GET_MODEM_PARAMETERS:
            printf("Modem parameters received - Mix Gain: %d, IF Gain: %d, Threshold: %d\n", raw[5], raw[6], (raw[7] << 8) + raw[8]);
            break;
        case COMMAND_SET_MODEM_PARAMETERS:
            printf("Modem parameters have been successfully changed.\n");
            break;
        case COMMAND_SET_FREQUENCY_HOPPING:
            printf("Frequency hopping has been successfully enabled.\n");
            break;
        default:
            printf("Unknown response.\n");
            break;
        }
    } else {
        printf("Unknown packet type.\n");
    }
}

void reset_reader_variables(int *data_buffer_size, int *packet_size) {
    *data_buffer_size = 0;
    *packet_size = 0;
}

static void uart_event_task(void *pvParameters) {
    uart_event_t event;
    int packet_size = 0;
    int packet_count = 0;
    int header_index;
    int processed;
    for (;;) {
        // Waiting for UART event.
        if (xQueueReceive(uart0_queue, (void *)&event, (TickType_t)portMAX_DELAY)) {
            switch (event.type) {
            case UART_DATA:
                packet_count++;
                // ESP_LOGI(TAG, "[UART DATA]: %d", event.size);
                uart_read_bytes(EX_UART_NUM, data_buffer + data_buffer_size, event.size, portMAX_DELAY);
                // ESP_LOGI(TAG, "Data buffer size: %d", data_buffer_size);
                data_buffer_size += event.size;
                // ESP_LOGI(TAG, "Data buffer size...: %d", data_buffer_size);
                processed = 0;
                // ESP_LOGI(TAG, "Processed");
                header_index = -1;
                // ESP_LOGI(TAG, "Header index");
                while ((header_index = search_for_bytes(data_buffer, data_buffer_size, (uint8_t *)"\xBB", 1, processed)) >= 0) {
                    // ESP_LOGI(TAG, "Header index12312312: %d", header_index);
                    vTaskDelay(1);
                    if (processed >= data_buffer_size) {
                        break;
                    }
                    if (header_index < 0) {
                        break;
                    }
                    if (data_buffer_size - header_index < 5) { // we don't have enough bytes for packet length
                        ESP_LOGI(TAG, "Not enough bytes for packet length");
                        break;
                    }

                    packet_size = ((data_buffer[header_index + 3] << 8) + data_buffer[header_index + 4]) + 7;
                    if (packet_size > MAX_PAYLOAD_SIZE) {
                        ESP_LOGI(TAG, "Packet too large");
                        ESP_LOG_BUFFER_HEXDUMP("DATA", data_buffer, packet_size, ESP_LOG_INFO);
                        processed = header_index + 1;
                        continue;
                    }

                    if (data_buffer_size - header_index < packet_size) {
                        ESP_LOGI(TAG, "Not enough bytes for packet");
                        break;
                    }

                    if (validate_packet(data_buffer + header_index, packet_size)) {
                        decode(data_buffer + header_index);
                        processed = header_index + packet_size;
                    } else {
                        ESP_LOGI(TAG, "Invalid packet");
                        processed = header_index + 1;
                    }
                }
                // print header index
                // ESP_LOGI(TAG, "Header index: %d", header_index);
                // ESP_LOGI(TAG, "Processed: %d", processed);
                // if header isn't -1, move the remaining bytes to the beginning of the buffer
                if (data_buffer_size - processed > 0) {
                    // ESP_LOGI(TAG, "Data buffer size 1: %d", data_buffer_size);
                    memmove(data_buffer, data_buffer + processed, data_buffer_size - processed);
                    data_buffer_size -= header_index;
                    // ESP_LOGI(TAG, "Data buffer size 2: %d", data_buffer_size);
                } else {
                    data_buffer_size = 0;
                    // ESP_LOGI(TAG, "Data buffer cleared");
                }
                break;
            // Event of HW FIFO overflow detected
            case UART_FIFO_OVF:
                ESP_LOGI(TAG, "hw fifo overflow");
                // If fifo overflow happened, you should consider adding flow control for your application.
                // The ISR has already reset the rx FIFO,
                // As an example, we directly flush the rx buffer here in order to read more data.
                uart_flush_input(EX_UART_NUM);
                xQueueReset(uart0_queue);
                break;
            // Event of UART ring buffer full
            case UART_BUFFER_FULL:
                ESP_LOGI(TAG, "ring buffer full");
                // If buffer full happened, you should consider increasing your buffer size
                // As an example, we directly flush the rx buffer here in order to read more data.
                uart_flush_input(EX_UART_NUM);
                xQueueReset(uart0_queue);
                break;
            // Event of UART RX break detected
            case UART_BREAK:
                ESP_LOGI(TAG, "uart rx break");
                break;
            // Event of UART parity check error
            case UART_PARITY_ERR:
                ESP_LOGI(TAG, "uart parity error");
                break;
            // Event of UART frame error
            case UART_FRAME_ERR:
                ESP_LOGI(TAG, "uart frame error");
                break;
            // Others
            default:
                ESP_LOGI(TAG, "uart event type: %d", event.type);
                break;
            }
        }
    }
    vTaskDelete(NULL);
}

void scan_timer_callback() {
    ESP_LOGI(TAG, "Stopping RFID scan");
    for (int i = 0; i < 5; i++) {
        stop_scan(400);
    }
    ESP_ERROR_CHECK(esp_event_post(RFID_EVENT, RFID_EVENT_SCAN_FINISHED, NULL, 0, 0));
}

void scan_tags(int duration) {
    // Create a task to handler UART event from ISR
    ESP_LOGI(TAG, "Starting RFID scan");
    start_scan(400);
    esp_timer_handle_t timer;
    const esp_timer_create_args_t timer_args = {
        .callback = &scan_timer_callback,
        .name = "periodic"};
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &timer));
    ESP_ERROR_CHECK(esp_timer_start_once(timer, duration * 1000000));
}

struct hashmap *rfid_init(void) {
    esp_log_level_set(TAG, ESP_LOG_INFO);

    /* Configure parameters of an UART driver,
     * communication pins and install the driver */
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    // Install UART driver, and get the queue.
    uart_driver_install(EX_UART_NUM, BUF_SIZE * 2, BUF_SIZE * 2, 20, &uart0_queue, 0);
    uart_param_config(EX_UART_NUM, &uart_config);

    // Set UART log level
    esp_log_level_set(TAG, ESP_LOG_INFO);
    // Set UART pins (using UART0 default pins ie no changes.)
    uart_set_pin(EX_UART_NUM, 14, 34, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);

    xTaskCreate(uart_event_task, "uart_event_task", 4096, NULL, 12, NULL);
    setup_rfid_module();

    // create a new hash map where each item is a `struct tag`. The second
    // argument is the initial capacity. The third and fourth arguments are
    // optional seeds that are passed to the following hash function.
    tag_map = hashmap_new(sizeof(struct tag), 0, 0, 0,
                          tag_hash, tag_compare, NULL, NULL);

    return tag_map;
}