
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


export function getOvertimePercentage(date: Date, isEmployee: boolean): number {
    const hour = date.getHours();
    const minute = date.getMinutes();
    const isWeekday = date.getDay() > 0 && date.getDay() < 6;
    const isSaturday = date.getDay() === 6;
    const isHoliday = isPublicHoliday(date) || date.getDay() === 0; // Including Sundays
    const isStandardHour = (hour > 9 || (hour === 9 && minute >= 0)) && (hour < 18 || (hour === 18 && minute === 0));
    const isNightHour = hour >= 22 || hour < 6;

    if (isEmployee) {
        if (isHoliday) {
            return isNightHour ? 50 : 30;
        } else if (isNightHour) {
            return 50;
        } else if (isSaturday) {
            return 15;
        } else if (isWeekday && !isStandardHour) {
            return 15;
        } else {
            return 0;
        }
    } else {
        return isHoliday ? 30 : 0;
    }
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

export function getOvertimeHoursCount(tasksOvertime: Record<number, number>): number {
    let overtimeHours = 0;
    for (const overtimePercent in tasksOvertime) {
        if (overtimePercent !== "0" && tasksOvertime[overtimePercent]) {
            overtimeHours += tasksOvertime[overtimePercent];
        }
    }
    return overtimeHours;
}