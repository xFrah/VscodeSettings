import { precisionRound } from "./dashboard_reports.resolver";

export function isPublicHoliday(date: Date): boolean {
    const year = date.getFullYear().toString();
    const monthDay = `${date.getMonth() + 1}`.padStart(2, '0') + `-${date.getDate()}`.padStart(2, '0');
    const holidays = [
        `01-01-${year}`,
        `01-06-${year}`,
        `04-25-${year}`,
        `05-01-${year}`,
        `06-02-${year}`,
        `08-15-${year}`,
        `11-01-${year}`,
        `12-08-${year}`,
        `12-25-${year}`,
        `12-26-${year}`
    ];
    return holidays.includes(monthDay + `-${year}`);
}


function getOvertimePercentage(date: Date, isEmployee: boolean): number {
    const hour = date.getHours();
    const isWeekday = date.getDay() > 0;
    const isSaturday = date.getDay() === 6;
    const isHoliday = isPublicHoliday(date) || date.getDay() === 0; // Including Sundays
    const isStandardHour = hour >= 9 && hour < 18;
    const isNightHour = hour >= 22 || hour < 6;

    // For employees, check if it's a holiday or night hour first
    if (isEmployee) {
        if (isHoliday) {
            // All hours during public holidays and Sundays are at 30%, except night hours
            return isNightHour ? 50 : 30;
        } else if (isNightHour) {
            // Night hours on standard weekdays and Saturdays are at 50%
            return 50;
        } else if (isSaturday) {
            // All hours during Saturdays are at 15%
            return 15;
        } else if (isWeekday && !isStandardHour) {
            // Weekday hours outside of standard hours are at 15%
            return 15;
        } else {
            // Hours within standard work hours on weekdays are not subject to overtime
            return 0;
        }
    } else {
        // For freelancers, holidays have 30% overtime for all hours
        return isHoliday ? 30 : 0;
    }
}

export function getOvertimeHoursCount(tasksOvertime: Record<number, number>): number {
    let overtimeHours = 0;
    for (const overtimePercent in tasksOvertime) {
        if (overtimePercent !== "0" && tasksOvertime[overtimePercent]) {
            overtimeHours += tasksOvertime[overtimePercent];
        }
    }
    return overtimeHours;
}

function calculateHourPercentage(minutes: number): number {
    return precisionRound(minutes / 60, 2);
}

export function correctHeadAndTail(startDateTime: string, endDateTime: string, overtimeMap: Map<string, number>, occurrences: Record<number, number>): Record<number, number> {
    const start = new Date(startDateTime);
    const end = new Date(endDateTime);
    if (overtimeMap.size === 0) {
        return occurrences;
    }
    // console.log("Start occurrences", occurrences);
    const firstElement = overtimeMap.entries().next().value[1];
    // if difference is under 60 minutes, set every occurrence to 0 except the first one, which is set to the percentage of minutes
    if ((end.getTime() - start.getTime()) / 60000 < 60 || overtimeMap.size === 1) {
        occurrences[firstElement] = calculateHourPercentage((end.getTime() - start.getTime()) / 60000);
        return occurrences;
    }

    console.log("asd", occurrences);
    const lastElement = Array.from(overtimeMap.entries()).pop()[1];
    // get minutes between start hour and next o'clock
    const minutesLate = start.getMinutes();
    // get minutes between end hour and previous o'clock
    const minutesOvertime = end.getMinutes();

    occurrences[firstElement] -= calculateHourPercentage(minutesLate);
    occurrences[lastElement] -= (1 - calculateHourPercentage(minutesOvertime));

    console.log("Task start and end time", start, end, minutesLate, calculateHourPercentage(minutesLate), minutesOvertime, (1 - calculateHourPercentage(minutesOvertime)), "Occurrences", occurrences)

    return occurrences
}

export function calculateHourlyOvertimePercentage(
    startDateTime: string,
    endDateTime: string,
    isEmployee: boolean // Added parameter to distinguish between employee and freelancer
): Map<string, number> {
    const start = new Date(startDateTime);
    const end = new Date(endDateTime);
    start.setMinutes(0);
    start.setSeconds(0);
    start.setMilliseconds(0);
    end.setMinutes(0);
    end.setSeconds(0);
    end.setMilliseconds(0);
    const overtimeMap = new Map<string, number>();
    // console.log("Task start and end time", start, end, "isEmployee", isEmployee)

    for (
        let current = new Date(start);
        current <= end;
        current.setHours(current.getHours() + 1)
    ) {
        console.log("Current", current)
        const overtimePercentage = getOvertimePercentage(current, isEmployee);
        overtimeMap.set(
            `${current.getHours()}`.padStart(2, '0') + ':00',
            overtimePercentage
        );
    }

    return overtimeMap;
}

export function countOvertimeOccurrences(overtimeMap: Map<string, number>): Record<number, number> {
    const overtimeCounts: Record<number, number> = { 0: 0, 15: 0, 30: 0, 50: 0 };
    ;

    overtimeMap.forEach((percentage) => {
        if (overtimeCounts[percentage]) {
            overtimeCounts[percentage]++;
        } else {
            overtimeCounts[percentage] = 1;
        }
    });

    return overtimeCounts;
}