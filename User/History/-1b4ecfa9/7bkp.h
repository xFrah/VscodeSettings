#include "esp_event.h"
#include "esp_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

extern esp_timer_handle_t g_timer;           // the periodic timer object

// Declare an event base
ESP_EVENT_DECLARE_BASE(RFID_EVENTS);        // declaration of the timer events family

enum {                                       // declaration of the specific events under the timer event family
    TIMER_EVENT_STARTED,                     // raised when the timer is first started
    TIMER_EVENT_EXPIRY,                      // raised when a period of the timer has elapsed
    TIMER_EVENT_STOPPED                      // raised when the timer has been stopped
};

ESP_EVENT_DECLARE_BASE(TASK_EVENTS);         // declaration of the task events family

enum {
    TASK_ITERATION_EVENT,                    // raised during an iteration of the loop within the task
};

#ifdef __cplusplus
}
#endif