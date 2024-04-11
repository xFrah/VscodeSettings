#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_event.h"
#include "esp_flash.h" // V5
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_netif_sntp.h"
#include "esp_sntp.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "hashmap.h"
#include "lwip/ip_addr.h"
#include "mqtt_client.h"
#include "nfc_events.h"
#include "nvs_flash.h"
#include "rfid_events.h"
#include "sdkconfig.h"
#include <esp_chip_info.h>
#include <stdio.h>

#include "ds3231.h"
#include "lis3mdl.h"
#include "m100.h"
#include "pn532.h"
#include <esp_sleep.h>
#include <flashdb.h>

#define __PRITS "ld"

struct env_status {
    int temp;
    int humi;
};

static bool query_cb(fdb_tsl_t tsl, void *arg);
static bool query_by_time_cb(fdb_tsl_t tsl, void *arg);
static bool set_status_cb(fdb_tsl_t tsl, void *arg);

#define WIFI_SSID (CONFIG_WIFI_SSID)
#define WIFI_PASSWORD (CONFIG_WIFI_PASSWORD)

#define MQTT_URI (CONFIG_BROKER_URI)

/* The event group allows multiple bits for each event, but we only care about two events:
 * - we are connected to the AP with an IP
 * - we failed to connect after the maximum amount of retries */
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1

#define ESP_MAXIMUM_RETRY 5

#define MQTT_MESSAGE_BUFFER_SIZE 4096

extern const uint8_t mqtt_eclipseprojects_io_pem_start[] asm("_binary_mqtt_cert_pem_start");
extern const uint8_t mqtt_eclipseprojects_io_pem_end[] asm("_binary_mqtt_cert_pem_end");

static struct hashmap *tag_hashmap;

static const char *TAG = "LOGS";
static const char *FDB_LOG_TAG = "FDB";

static char mqtt_message_buffer[MQTT_MESSAGE_BUFFER_SIZE];

esp_mqtt_client_handle_t mqtt_client_h;

/* FreeRTOS event group to signal when we are connected*/
static EventGroupHandle_t s_wifi_event_group;

static int s_retry_num = 0;

static uint32_t boot_count = 0;
static time_t boot_time[10] = {0, 1, 2, 3};

/* default KV nodes */
static struct fdb_default_kv_node default_kv_table[] = {
    {"username", "armink", 0},                       /* string KV */
    {"password", "123456", 0},                       /* string KV */
    {"boot_count", &boot_count, sizeof(boot_count)}, /* int type KV */
    {"boot_time", &boot_time, sizeof(boot_time)},    /* int array type KV */
};
/* KVDB object */
static struct fdb_kvdb kvdb = {0};
// /* Time series database object */
// struct fdb_tsdb tsdb = {0};

/* counts for simulated timestamp */
static int counts = 0;
static SemaphoreHandle_t s_lock = NULL;

static void lock(fdb_db_t db) {
    xSemaphoreTake(s_lock, portMAX_DELAY);
}

static void unlock(fdb_db_t db) {
    xSemaphoreGive(s_lock);
}

// static fdb_time_t get_time(void) {
//     /* Using the counts instead of timestamp.
//      * Please change this function to return RTC time.
//      */
//     return ++counts;
// }

// void tsdb_sample(fdb_tsdb_t tsdb) {
//     // struct fdb_blob blob;

//     ESP_LOGI(TAG, "==================== tsdb_sample ====================");

//     // { /* APPEND new TSL (time series log) */
//     //     struct env_status status;

//     //     /* append new log to TSDB */
//     //     status.temp = 36;
//     //     status.humi = 85;
//     //     fdb_tsl_append(tsdb, fdb_blob_make(&blob, &status, sizeof(status)));
//     //     ESP_LOGI(FDB_LOG_TAG, "append the new status.temp (%d) and status.humi (%d)\n", status.temp, status.humi);

//     //     status.temp = 38;
//     //     status.humi = 90;
//     //     fdb_tsl_append(tsdb, fdb_blob_make(&blob, &status, sizeof(status)));
//     //     ESP_LOGI(FDB_LOG_TAG, "append the new status.temp (%d) and status.humi (%d)\n", status.temp, status.humi);
//     // }

//     { /* QUERY the TSDB */
//         /* query all TSL in TSDB by iterator */
//         fdb_tsl_iter(tsdb, query_cb, tsdb);
//     }

//     { /* QUERY the TSDB by time */
//         /* prepare query time (from 1970-01-01 00:00:00 to 2020-05-05 00:00:00) */
//         struct tm tm_from = {.tm_year = 1970 - 1900, .tm_mon = 0, .tm_mday = 1, .tm_hour = 0, .tm_min = 0, .tm_sec = 0};
//         struct tm tm_to = {.tm_year = 2020 - 1900, .tm_mon = 4, .tm_mday = 5, .tm_hour = 0, .tm_min = 0, .tm_sec = 0};
//         time_t from_time = mktime(&tm_from), to_time = mktime(&tm_to);
//         size_t count;
//         /* query all TSL in TSDB by time */
//         fdb_tsl_iter_by_time(tsdb, from_time, to_time, query_by_time_cb, tsdb);
//         /* query all FDB_TSL_WRITE status TSL's count in TSDB by time */
//         count = fdb_tsl_query_count(tsdb, from_time, to_time, FDB_TSL_WRITE);
//         ESP_LOGI(FDB_LOG_TAG, "query count is: %zu", count);
//     }

//     { /* SET the TSL status */
//         /* Change the TSL status by iterator or time iterator
//          * set_status_cb: the change operation will in this callback
//          *
//          * NOTE: The actions to modify the state must be in order.
//          *       like: FDB_TSL_WRITE -> FDB_TSL_USER_STATUS1 -> FDB_TSL_DELETED -> FDB_TSL_USER_STATUS2
//          *       The intermediate states can also be ignored.
//          *       such as: FDB_TSL_WRITE -> FDB_TSL_DELETED
//          */
//         fdb_tsl_iter(tsdb, set_status_cb, tsdb);
//     }

//     ESP_LOGI(FDB_LOG_TAG, "===========================================================");
// }

// static bool query_cb(fdb_tsl_t tsl, void *arg) {
//     struct fdb_blob blob;
//     struct env_status status = {0};
//     fdb_tsdb_t db = arg;

//     fdb_blob_read((fdb_db_t)db, fdb_tsl_to_blob(tsl, fdb_blob_make(&blob, &status, sizeof(status))));
//     ESP_LOGI(FDB_LOG_TAG, "[query_cb] queried a TSL: time: %" __PRITS ", temp: %d, humi: %d", tsl->time, status.temp, status.humi);

//     return false;
// }

// static bool query_by_time_cb(fdb_tsl_t tsl, void *arg) {
//     struct fdb_blob blob;
//     struct env_status status = {0};
//     fdb_tsdb_t db = arg;

//     fdb_blob_read((fdb_db_t)db, fdb_tsl_to_blob(tsl, fdb_blob_make(&blob, &status, sizeof(status))));
//     ESP_LOGI(FDB_LOG_TAG, "[query_by_time_cb] queried a TSL: time: %" __PRITS ", temp: %d, humi: %d", tsl->time, status.temp, status.humi);

//     return false;
// }

// static bool set_status_cb(fdb_tsl_t tsl, void *arg) {
//     fdb_tsdb_t db = arg;

//     ESP_LOGI(FDB_LOG_TAG, "set the TSL (time %" __PRITS ") status from %d to %d", tsl->time, tsl->status, FDB_TSL_USER_STATUS1);
//     fdb_tsl_set_status(db, tsl, FDB_TSL_USER_STATUS1);

//     return false;
// }

/*
 * @brief Event handler registered to receive MQTT events
 *
 *  This function is called by the MQTT client event loop.
 *
 * @param handler_args user data registered to the event.
 * @param base Event base for the handler(always MQTT Base in this example).
 * @param event_id The id for the received event.
 * @param event_data The data for the event, esp_mqtt_event_handle_t.
 */
static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    ESP_LOGD(TAG, "Event dispatched from event loop base=%s, event_id=%" PRIi32, base, event_id);
    esp_mqtt_event_handle_t event = event_data;
    int msg_id;
    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
        msg_id = esp_mqtt_client_subscribe(mqtt_client_h, "/topic/qos0", 0);
        ESP_LOGI(TAG, "sent subscribe successful, msg_id=%d", msg_id);
        break;
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
        break;
    case MQTT_EVENT_SUBSCRIBED:
        ESP_LOGI(TAG, "MQTT_EVENT_SUBSCRIBED, msg_id=%d", event->msg_id);
        msg_id = esp_mqtt_client_publish(mqtt_client_h, "/topic/qos0", "data", 0, 0, 0);
        ESP_LOGI(TAG, "sent publish successful, msg_id=%d", msg_id);
        break;
    case MQTT_EVENT_UNSUBSCRIBED:
        ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED, msg_id=%d", event->msg_id);
        break;
    case MQTT_EVENT_PUBLISHED:
        ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED, msg_id=%d", event->msg_id);
        break;
    case MQTT_EVENT_DATA:
        ESP_LOGI(TAG, "MQTT_EVENT_DATA");
        printf("TOPIC=%.*s\r\n", event->topic_len, event->topic);
        printf("DATA=%.*s\r\n", event->data_len, event->data);
        if (strncmp(event->data, "send binary please", event->data_len) == 0) {
            ESP_LOGI(TAG, "Sending the binary");
        }
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGI(TAG, "MQTT_EVENT_ERROR");
        if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
            // ESP_LOGI(TAG, "Last error code reported from esp-tls: 0x%x", event->error_handle->esp_tls_last_esp_err);
            // ESP_LOGI(TAG, "Last tls stack error number: 0x%x", event->error_handle->esp_tls_stack_err);
            ESP_LOGI(TAG, "Last captured errno : %d (%s)", event->error_handle->esp_transport_sock_errno,
                     strerror(event->error_handle->esp_transport_sock_errno));
        } else if (event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED) {
            ESP_LOGI(TAG, "Connection refused error: 0x%x", event->error_handle->connect_return_code);
        } else {
            ESP_LOGW(TAG, "Unknown error type: 0x%x", event->error_handle->error_type);
        }
        break;
    default:
        ESP_LOGI(TAG, "Other event id:%d", event->event_id);
        break;
    }
}

void time_sync_notification_cb(struct timeval *tv) {
    const ip_addr_t *ip = esp_sntp_getserver(0);
    const ip_addr_t *ip2 = esp_sntp_getserver(1);
    char ip_addr_str[16];
    char ip2_addr_str[16];
    ip4addr_ntoa_r((const ip4_addr_t *)ip, ip_addr_str, sizeof(ip_addr_str));
    ip4addr_ntoa_r((const ip4_addr_t *)ip2, ip2_addr_str, sizeof(ip2_addr_str));
    ESP_LOGI(TAG, "Notification of a time synchronization event, time: %llu, ip: %s, ip2: %s", tv->tv_sec, ip_addr_str, ip2_addr_str);
    // convert time to readable format
    struct tm timeinfo;
    localtime_r(&tv->tv_sec, &timeinfo);
    char strftime_buf[64];
    strftime(strftime_buf, sizeof(strftime_buf), "%c", &timeinfo);
    ESP_LOGI(TAG, "The current date/time in New York is: %s", strftime_buf);
}

static void mqtt_init() {
    const esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address.uri = MQTT_URI,
            // .verification.certificate = (const char *)mqtt_eclipseprojects_io_pem_start
        },
    };
    ESP_LOGI(TAG, "MQTT URI: %s", mqtt_cfg.broker.address.uri);

    ESP_LOGI(TAG, "[APP] Free memory: %" PRIu32 " bytes", esp_get_free_heap_size());
    mqtt_client_h = esp_mqtt_client_init(&mqtt_cfg);
    /* The last argument may be used to pass data to the event handler, in this example mqtt_event_handler */
    esp_mqtt_client_register_event(mqtt_client_h, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(mqtt_client_h);
}

int compose_mqtt_message(struct tag *tags, int tag_count) {
    mqtt_message_buffer[0] = '{';
    char *current_pointer = mqtt_message_buffer + 1;
    int remaining = MQTT_MESSAGE_BUFFER_SIZE - (current_pointer - mqtt_message_buffer);
    int written;
    written = snprintf(current_pointer, remaining, "\"tags\":[");
    if (written < 0) {
        ESP_LOGE(TAG, "Failed to write to message buffer");
        return -1;
    }
    current_pointer += written;

    for (size_t i = 0; i < tag_count; i++) {
        remaining = MQTT_MESSAGE_BUFFER_SIZE - (current_pointer - mqtt_message_buffer);
        written = snprintf(current_pointer, remaining, "\"%s\":%d,", tags[i].id, tags[i].timestamp);
        if (written < 0) {
            ESP_LOGE(TAG, "Failed to write to message buffer");
            break;
        }
        current_pointer += written;
    }
    *(current_pointer - 1) = ']'; // overwrite the last comma
    *(current_pointer++) = ',';   // add a comma

    // add timestamp from sntp
    remaining = MQTT_MESSAGE_BUFFER_SIZE - (current_pointer - mqtt_message_buffer);
    time_t now;
    struct tm timeinfo;
    char strftime_buf[64];

    time(&now);                   // Get current time
    localtime_r(&now, &timeinfo); // Convert to local time format
    strftime(strftime_buf, sizeof(strftime_buf), "%Y-%m-%d %H:%M:%S", &timeinfo);
    written = snprintf(current_pointer, remaining, "\"timestamp\":\"%s\"}", strftime_buf);
    if (written < 0) {
        ESP_LOGE(TAG, "Failed to write to message buffer");
        return -1;
    }
    current_pointer += written;
    return current_pointer - mqtt_message_buffer;
}

void get_tags_from_hashmap(struct tag *tags, int tag_count) {
    size_t iter = 0;
    int i = 0;
    void *item;
    while (hashmap_iter(tag_hashmap, &iter, &item)) {
        if (i >= tag_count) { // Ensure we don't write past the end of the tags array
            ESP_LOGE(TAG, "Iterator exceeds expected count");
            break;
        }
        struct tag *tag_ptr = (struct tag *)item; // Explicitly showing the two-step process
        if (tag_ptr != NULL) {                    // Check if the tag pointer is valid
            ESP_LOGI(TAG, "Tag %d: %s (%d)", i, tag_ptr->id, tag_ptr->timestamp);
            tags[i] = *tag_ptr; // Copy the struct tag into the tags array
        } else {
            ESP_LOGE(TAG, "Null tag pointer encountered at iter=%zu", iter);
        }
        i++;
    }
}

static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_sntp_config_t config = ESP_NETIF_SNTP_DEFAULT_CONFIG("ntp3.ntp-servers.net");
        config.start = false;                           // start the SNTP service explicitly (after connecting)
        config.server_from_dhcp = true;                 // accept the NTP offers from DHCP server
        config.renew_servers_after_new_IP = true;       // let esp-netif update the configured SNTP server(s) after receiving the DHCP lease
        config.index_of_first_server = 1;               // updates from server num 1, leaving server 0 (from DHCP) intact
        config.ip_event_to_renew = IP_EVENT_STA_GOT_IP; // IP event on which we refresh the configuration
        config.sync_cb = time_sync_notification_cb;     // only if we need the notification function

        esp_sntp_setservername(1, "ntp4.ntp-servers.net");

        ESP_ERROR_CHECK(esp_netif_sntp_init(&config));
        ESP_ERROR_CHECK(esp_netif_sntp_start());
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < ESP_MAXIMUM_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "retry to connect to the AP");
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
        ESP_LOGI(TAG, "connect to the AP fail");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == RFID_EVENT && event_id == RFID_EVENT_SCAN_FINISHED) { // make it have its own event handler
        ESP_LOGI(TAG, "RFID_EVENT_SCAN_FINISHED");
        int tag_count = hashmap_count(tag_hashmap);
        if (tag_count > 32) {
            tag_count = 32;
        }
        ESP_LOGI(TAG, "Tag count: %d", tag_count);
        struct tag *tags = malloc(sizeof(struct tag) * tag_count);
        if (!tags) {
            ESP_LOGE(TAG, "Failed to allocate memory for tags");
            hashmap_clear(tag_hashmap, false);
            return;
        }
        get_tags_from_hashmap(tags, tag_count);
        int total_length = compose_mqtt_message(tags, tag_count);
        if (esp_mqtt_client_publish(mqtt_client_h, "/topic/qos0", mqtt_message_buffer, total_length, 1, 0) < 0) { // enable skip if disconnected in config
            ESP_LOGE(TAG, "Failed to publish to MQTT broker");
            struct fdb_blob blob;
            for (size_t i = 0; i < tag_count; i++) {
                ESP_LOGI(TAG, "Tag %zu: %s", i, tags[i].id);
                fdb_kv_set_blob(&kvdb, tags[i].id, fdb_blob_make(&blob, &tags->timestamp, sizeof(tags->timestamp)));
            }
            struct fdb_kv_iterator iterator;
            fdb_kv_t cur_kv;
            size_t data_size;
            uint8_t *data_buf;

            fdb_kv_iterator_init(&kvdb, &iterator);

            while (fdb_kv_iterate(&kvdb, &iterator)) {
                cur_kv = &(iterator.curr_kv);
                data_size = (size_t)cur_kv->value_len;
                data_buf = (uint8_t *)malloc(data_size);
                if (data_buf == NULL) {
                    ESP_LOGI(TAG, "Error: malloc failed.\n");
                    break;
                }
                fdb_blob_read((fdb_db_t)&kvdb, fdb_kv_to_blob(cur_kv, fdb_blob_make(&blob, data_buf, data_size)));
                ESP_LOGI(TAG, "Value: %s", data_buf); // still don't know how to get the key
                free(data_buf);
            }
        }
        free(tags);
    } else if (event_base == NFC_EVENT && event_id == NFC_EVENT_TAG_FOUND) { // make it have its own event handler
        ESP_LOGI(TAG, "NFC_EVENT_TAG_FOUND");
        // query the credentials db for the tag
    }
}

void wifi_init(void) {
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // esp_event_handler_instance_t instance_any_id;
    // esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta finished.");

    /* Waiting until either the connection is established (WIFI_CONNECTED_BIT) or connection failed for the maximum
     * number of re-tries (WIFI_FAIL_BIT). The bits are set by event_handler() (see above) */
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           pdFALSE,
                                           pdFALSE,
                                           portMAX_DELAY);

    /* xEventGroupWaitBits() returns the bits before the call returned, hence we can test which event actually
     * happened. */
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "connected to ap SSID:%s password:%s",
                 WIFI_SSID, WIFI_PASSWORD);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGI(TAG, "Failed to connect to SSID:%s, password:%s",
                 WIFI_SSID, WIFI_PASSWORD);
    } else {
        ESP_LOGE(TAG, "UNEXPECTED EVENT");
    }
}

int flashdb_init(void) {
    fdb_err_t result;

    if (s_lock == NULL) {
        s_lock = xSemaphoreCreateCounting(1, 1);
        assert(s_lock != NULL);
    }

    { /* KVDB Sample */
        struct fdb_default_kv default_kv;

        default_kv.kvs = default_kv_table;
        default_kv.num = sizeof(default_kv_table) / sizeof(default_kv_table[0]);
        /* set the lock and unlock function if you want */
        fdb_kvdb_control(&kvdb, FDB_KVDB_CTRL_SET_LOCK, lock);
        fdb_kvdb_control(&kvdb, FDB_KVDB_CTRL_SET_UNLOCK, unlock);
        /* Key-Value database initialization
         *
         *       &kvdb: database object
         *       "env": database name
         * "fdb_kvdb1": The flash partition name base on FAL. Please make sure it's in FAL partition table.
         *              Please change to YOUR partition name.
         * &default_kv: The default KV nodes. It will auto add to KVDB when first initialize successfully.
         *        NULL: The user data if you need, now is empty.
         */
        result = fdb_kvdb_init(&kvdb, "env", "fdb_kvdb1", &default_kv, NULL);

        if (result != FDB_NO_ERR) {
            return -1;
        }

        ESP_LOGI(FDB_LOG_TAG, "KVDB init success");
    }

    // { /* TSDB Sample */
    //     /* set the lock and unlock function if you want */
    //     fdb_tsdb_control(&tsdb, FDB_TSDB_CTRL_SET_LOCK, lock);
    //     fdb_tsdb_control(&tsdb, FDB_TSDB_CTRL_SET_UNLOCK, unlock);
    //     /* Time series database initialization
    //      *
    //      *       &tsdb: database object
    //      *       "log": database name
    //      * "fdb_tsdb1": The flash partition name base on FAL. Please make sure it's in FAL partition table.
    //      *              Please change to YOUR partition name.
    //      *    get_time: The get current timestamp function.
    //      *         128: maximum length of each log
    //      *        NULL: The user data if you need, now is empty.
    //      */
    //     result = fdb_tsdb_init(&tsdb, "log", "fdb_tsdb1", get_time, 128, NULL);
    //     /* read last saved time for simulated timestamp */
    //     fdb_tsdb_control(&tsdb, FDB_TSDB_CTRL_GET_LAST_TIME, &counts);

    //     if (result != FDB_NO_ERR) {
    //         return -1;
    //     }

    //     ESP_LOGI(FDB_LOG_TAG, "TSDB init success");
    //     tsdb_sample(&tsdb);
    // }

    return 0;
}

void custom_events_init(void) {
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_event_handler_instance_register(RFID_EVENT,
                                                        RFID_EVENT_SCAN_FINISHED,
                                                        &event_handler,
                                                        NULL,
                                                        NULL));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(NFC_EVENT,
                                                        NFC_EVENT_TAG_FOUND,
                                                        &event_handler,
                                                        NULL,
                                                        NULL));
}

void app_main(void) {
    /* Print chip information */
    esp_chip_info_t chip_info;
    uint32_t size_flash_chip;
    esp_chip_info(&chip_info);
    esp_flash_get_size(NULL, &size_flash_chip);
    ESP_LOGI(TAG, "This is ESP32 chip with %d CPU cores, silicon revision %d, %lxB flash, IDF version: %s",
             chip_info.cores, chip_info.revision, size_flash_chip, esp_get_idf_version());

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    flashdb_init();
    custom_events_init();
    tag_hashmap = rfid_init();
    scan_tags(10);
    nfc_init();

    wifi_init();
    mqtt_init();
}
