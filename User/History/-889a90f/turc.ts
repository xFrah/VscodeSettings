import moment from "moment";
import { prisma } from "../db";
import { FUNCTION, QUERY, empTrackerLogs } from "../log";
import { precisionRound } from "./dashboard_reports.resolver";
import { headingStyle, uploadFileToAWS } from "./reports.resolver";
import { calculateTimeOfTask } from "./tasks.resolver";
const XlsxPopulate = require("xlsx-populate");

export function createAWS_S3_URL(key) {
    empTrackerLogs(`${FUNCTION} createAWS_S3_URL `);
    return `https://${process.env.S3_BUCKET_NAME}.s3.amazonaws.com/${key}`;
}

function isPublicHoliday(date: Date): boolean {
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
    const isWeekday = date.getDay() > 0 && date.getDay() < 6;
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

export async function sortDataForURI(data) {
    empTrackerLogs(`${FUNCTION} sortDataForURI`);
    const SORT_DATA = data.map((iteratorObj) => {
        const KEY = `${process.env.S3_BUCKET_FOLDER_NAME}/${encodeURIComponent(
            iteratorObj.file_name
        )}`;
        let fileUrl = null;
        if (iteratorObj.file_name) {
            fileUrl = createAWS_S3_URL(KEY);
        }
        return {
            ...iteratorObj,
            fileUrl,
        };
    });
    return SORT_DATA;
}

export async function workerDashboard(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} workerDashboard`);
    const FROM_DATE = args?.input?.fromDate;
    let TO_DATE = args?.input?.toDate;
    const WORKER_ID = args?.input?.profile_id || contextValue?.profile_id;
    if (!TO_DATE) {
        const NewEndDate = new Date(FROM_DATE);
        NewEndDate.setMinutes(1439);
        NewEndDate.setSeconds(59);
        TO_DATE = NewEndDate;
    }
    const result = {};
    const GET_WORKER = await prisma.profile.findFirst({
        where: { id: WORKER_ID, AND: { role_id: "4" } },
        include: { user: true },
    });
    const WORKER_NAME = `${GET_WORKER.user.first_name} ${GET_WORKER.user.last_name}`;
    result["worker_name"] = WORKER_NAME;
    result["hourly_wages"] = GET_WORKER?.hourly_wages;
    result["qualification"] = GET_WORKER?.qualification;

    let paramsforTask;
    let paramsforReceipts;
    if (!FROM_DATE) {
        paramsforTask = {
            orderBy: { start_time: "desc" },
            where: {
                deleted_at: { equals: null },
                AND: {
                    activity_worker: {
                        worker_id: WORKER_ID,
                    },
                },
            },
            include: {
                activity_worker: {
                    select: {
                        project_worker_id: {
                            select: {
                                worker_type: true,
                                hourly_wages: true,
                                user: { select: { first_name: true, last_name: true } },
                            },
                        },
                        project_activity_id: {
                            select: {
                                project_id: {
                                    select: {
                                        client: { select: { name: true } },
                                        name: true,
                                        project_id: true,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        };
        paramsforReceipts = {
            where: {
                deleted_at: { equals: null },
                AND: { created_by: { equals: WORKER_ID } },
            },
            include: {
                receipt_type: { select: { type: true } },
                attachment_created_by: {
                    select: { user: { select: { first_name: true, last_name: true } } },
                },
                activity_attachment_id: {
                    select: { project_id: { select: { project_id: true, client: { select: { name: true } } } } },
                },
            },
        };
    } else {
        paramsforTask = {
            orderBy: { start_time: "desc" },
            where: {
                deleted_at: { equals: null },
                AND: {
                    activity_worker: {
                        worker_id: WORKER_ID,
                    },
                    AND: {
                        start_time: { gte: FROM_DATE },
                        AND: { end_time: { lte: TO_DATE } },
                    },
                },
            },
            include: {
                activity_worker: {
                    select: {
                        project_worker_id: {
                            select: {
                                worker_type: true,
                                hourly_wages: true,
                                user: { select: { first_name: true, last_name: true } },
                            },
                        },
                        project_activity_id: {
                            select: {
                                project_id: {
                                    select: { client: true, name: true, project_id: true },
                                },
                            },
                        },
                    },
                },
            },
        };
        paramsforReceipts = {
            where: {
                deleted_at: { equals: null },
                AND: {
                    created_by: { equals: WORKER_ID },
                    AND: {
                        created_at: { gte: FROM_DATE },
                        AND: { created_at: { lte: TO_DATE } },
                    },
                },
            },
            include: {
                receipt_type: { select: { type: true } },
                attachment_created_by: {
                    select: { user: { select: { first_name: true, last_name: true } } },
                },
                activity_attachment_id: {
                    select: { project_id: { select: { project_id: true, client: { select: { name: true } } } } },
                },
            },
        };
    }

    // =====================T==A==S==K====================================

    const tasks: any = await prisma.activityWorkerTasks.findMany(paramsforTask);

    let totalTaskRawMinutes = 0;
    let totalOvertimeHours = 0;
    let allTasksOvertime: any = { 15: 0, 30: 0, 50: 0 };
    let projectGroupByHours: any = {};

    let allProjectTaskGroupBy = {};

    tasks.forEach((task) => {

        const projectName =
            task?.activity_worker?.project_activity_id?.project_id?.name;

        const PROJECT_ID =
            task?.activity_worker?.project_activity_id?.project_id?.project_id;

        const HOURLY_WAGES_OF_WORKER =
            task?.activity_worker?.project_worker_id?.hourly_wages;

        const WORKER_TYPE =
            task?.activity_worker?.project_worker_id?.worker_type;

        const CLIENT_NAME =
            task?.activity_worker?.project_activity_id?.project_id?.client?.name;

        const CREATED_BY = `${task?.activity_worker?.project_worker_id?.user?.first_name} ${task?.activity_worker?.project_worker_id?.user?.last_name}`;

        const taskDurationMin = calculateTimeOfTask(task?.start_time, task?.end_time)

        const overtimeMap = calculateHourlyOvertimePercentage(task?.start_time, task?.end_time, WORKER_TYPE === "EMPLOYEE");
        let tasksOvertime = countOvertimeOccurrences(overtimeMap);

        tasksOvertime = correctHeadAndTail(task?.start_time, task?.end_time, overtimeMap, tasksOvertime);
        // console.log("tasksOvertime", tasksOvertime);

        const taskOrdinaryHours = tasksOvertime[0];

        task["created_by"] = CREATED_BY;
        task["ordinary_hours"] = taskOrdinaryHours;
        task["client"] = CLIENT_NAME;
        task["project_name"] = projectName;
        // get sum of all overtime hours
        let overtimeHours = 0;
        for (const overtimePercent in tasksOvertime) {
            if (overtimePercent !== "0" && tasksOvertime[overtimePercent]) {
                overtimeHours += tasksOvertime[overtimePercent];
            }
        }
        task["overtime_hours"] = overtimeMap.size - taskOrdinaryHours;

        totalTaskRawMinutes += taskDurationMin;
        totalOvertimeHours += task["overtime_hours"];

        for (const overtimePercent in tasksOvertime) {
            if (overtimePercent !== "0" && tasksOvertime[overtimePercent]) {
                allTasksOvertime[overtimePercent] = precisionRound(
                    allTasksOvertime[overtimePercent] + tasksOvertime[overtimePercent],
                    2
                );
            }
        }

        task["hours_overtime_al_15"] = tasksOvertime[15];
        task["hours_overtime_al_30"] = tasksOvertime[30];
        task["hours_overtime_al_50"] = tasksOvertime[50];

        let TOTAL_COST = HOURLY_WAGES_OF_WORKER * taskOrdinaryHours;
        for (const overtimePercent in tasksOvertime) {
            if (overtimePercent !== "0" && tasksOvertime[overtimePercent]) {
                TOTAL_COST += (HOURLY_WAGES_OF_WORKER * tasksOvertime[overtimePercent]) * (1 + (parseInt(overtimePercent) / 100));
            }
        }
        console.log("TOTOL_COST", TOTAL_COST, "mins", taskDurationMin, "OrdHours", taskOrdinaryHours, "wage", HOURLY_WAGES_OF_WORKER, tasksOvertime);

        const projectHours = projectGroupByHours[projectName]
            ? projectGroupByHours[projectName] +
            taskOrdinaryHours +
            task["overtime_hours"]
            : taskOrdinaryHours + task["overtime_hours"];

        task["totalCost"] = TOTAL_COST;
        task["projectId"] = PROJECT_ID;
        projectGroupByHours[projectName] = projectHours;

        if (allProjectTaskGroupBy[projectName]) {
            allProjectTaskGroupBy[projectName]["ordinary_hours"] += taskOrdinaryHours;
            allProjectTaskGroupBy[projectName]["overtime_hours"] += task["overtime_hours"];
            allProjectTaskGroupBy[projectName]["totalCost"] = precisionRound(
                allProjectTaskGroupBy[projectName]["totalCost"] + TOTAL_COST,
                3
            );
        } else {
            allProjectTaskGroupBy[projectName];
            allProjectTaskGroupBy[projectName] = {
                client: CLIENT_NAME,
                project_name: projectName,
                ordinary_hours: taskOrdinaryHours,
                overtime_hours: task["overtime_hours"],
                totalCost: precisionRound(TOTAL_COST, 3),
                project_id: PROJECT_ID,
            };
        }
    });

    let allProjectTaskGroupByItems = [];
    let overAllTotalCostOfWorker = 0
    let overAllTotalHourOvertimeHours = 0
    for (const iterator in allProjectTaskGroupBy) {
        overAllTotalCostOfWorker += allProjectTaskGroupBy[iterator]['totalCost']
        overAllTotalHourOvertimeHours += allProjectTaskGroupBy[iterator]['overtime_hours']
        allProjectTaskGroupByItems.push(allProjectTaskGroupBy[iterator]);
    }

    totalTaskRawMinutes = parseFloat(totalTaskRawMinutes.toFixed(2));
    let duration = moment.duration(totalTaskRawMinutes, "minutes");
    let totalHours = parseFloat(duration.asHours().toFixed(2));

    result["project_details"] = allProjectTaskGroupByItems;
    result["tasks"] = tasks;

    result["totalTaskTime"] = totalHours;
    const projectGroupByHoursArray = [];
    for (const key in projectGroupByHours) {
        projectGroupByHoursArray.push({
            project_name: key,
            total_hours: projectGroupByHours[key],
        });
    }

    const allTasksOvertimeArray = [];
    let allTasksOvertimeTotal = 0;
    for (const key in allTasksOvertime) {
        allTasksOvertimeTotal += allTasksOvertime[key]
        allTasksOvertimeArray.push({
            overtimePercent: key,
            overtimeHours: allTasksOvertime[key],
        });
    }
    allTasksOvertimeTotal = overAllTotalHourOvertimeHours
    projectGroupByHours = projectGroupByHoursArray;
    allTasksOvertime = allTasksOvertimeArray;
    result["overview"] = {
        projectGroupByHours,
        allTasksOvertime,
        allTasksOvertimeTotal,
        overAllTotalCostOfWorker
    };



    // ===============R==E==C==E==I==P==T==============================

    const receipts: any = await prisma.activityAttachments.findMany(
        paramsforReceipts
    );


    const RECEIPT_DATA = await sortDataForURI(receipts);
    result["receipts"] = RECEIPT_DATA;
    let totalValueOfReceipt = 0;
    for (const receipt of RECEIPT_DATA) {
        let createdBy = `${receipt?.attachment_created_by?.user?.first_name} ${receipt?.attachment_created_by?.user?.last_name}`;
        receipt["created_by"] = createdBy;
        const projectId = receipt?.activity_attachment_id?.project_id?.project_id;
        receipt["project_id"] = projectId;
        receipt["client"] = receipt?.activity_attachment_id?.project_id?.client?.name
        receipt["receipt_type"] = receipt?.receipt_type?.type
        totalValueOfReceipt += parseFloat(receipt["receipt_value"]);
    }
    result["totalValueOfReceipt"] = precisionRound(totalValueOfReceipt, 3);

    return result;
}

export async function generateWorkerCSV(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} generateWorkerCSV`);
    try {
        const OVERALL_DETAILS: any = await workerDashboard("", args, "");

        const workbook = await XlsxPopulate.fromBlankAsync();

        //=========== File 1 Sheet 1 ===========
        const SHEET_1 = await workbook.sheet(0).name("Worker Details");
        const OVERALL_DETAILS_ARRAY = [];
        for (const iterator in OVERALL_DETAILS) {

            if (iterator == "worker_name") {
                OVERALL_DETAILS_ARRAY.push({
                    Name: "Worker Name",
                    Value: OVERALL_DETAILS[iterator],
                });
            } else if (iterator == "hourly_wages") {
                OVERALL_DETAILS_ARRAY.push({
                    Name: "Hourly Wages",
                    Value: parseFloat(OVERALL_DETAILS[iterator]),
                });
            } else if (iterator == "qualification") {
                OVERALL_DETAILS_ARRAY.push({
                    Name: "Qualification",
                    Value: OVERALL_DETAILS[iterator],
                });
            } else if (iterator == "totalTaskTime") {
                OVERALL_DETAILS_ARRAY.push({
                    Name: "Total Task Time",
                    Value: OVERALL_DETAILS[iterator],
                });
            } else if (iterator == "totalValueOfReceipt") {
                OVERALL_DETAILS_ARRAY.push({
                    Name: "Total Cost of Recipt",
                    Value: OVERALL_DETAILS[iterator],
                });
            }
            else if (iterator == "overview") {
                if (OVERALL_DETAILS[iterator]["overAllTotalCostOfWorker"]) {
                    OVERALL_DETAILS_ARRAY.push({
                        Name: "Total Cost of Task",
                        Value: OVERALL_DETAILS[iterator]["overAllTotalCostOfWorker"],
                    });
                }

            }
        }
        const SHEET_1_FIELDS = ["Name", "Value"];

        SHEET_1_FIELDS.forEach((field, index) => {
            headingStyle(SHEET_1, 1, index + 1, "990000");
            SHEET_1.cell(1, index + 1).value(field);
        });

        OVERALL_DETAILS_ARRAY.forEach((row, rowIndex) => {
            SHEET_1_FIELDS.forEach((field, columnIndex) => {
                SHEET_1.cell(rowIndex + 2, columnIndex + 1).value(row[field]);
            });
        });
        for (let columnIndex = 1; columnIndex <= SHEET_1_FIELDS.length; columnIndex++) {
            SHEET_1.column(columnIndex).width(
                Math.max(15, ...OVERALL_DETAILS_ARRAY.map(row => (row[SHEET_1_FIELDS[columnIndex - 1]]?.toString() || '').length))
            );
        }

        //=========== File 1 Sheet 2 ===========

        const PROJECT_TASKS = OVERALL_DETAILS["tasks"];

        const SHEET_2 = await workbook.addSheet("Project Tasks");

        let TASK_FIELDS_TEMP = [
            { name: "projectId", lable: "Project id" },
            { name: "client", lable: "Client" },
            { name: "project_name", lable: "Project Name" },
            { name: "name", lable: "Task" },
            { name: "created_at", lable: "Created On" },
            { name: "start_time", lable: "Start time" },
            { name: "end_time", lable: "End Time" },
            { name: "ordinary_hours", lable: "Ordinary Hours" },
            { name: "hours_overtime_al_15", lable: "Hours Overtime AL 15%" },
            { name: "hours_overtime_al_30", lable: "Hours Overtime AL 30%" },
            { name: "hours_overtime_al_50", lable: "Hours Overtime AL 50%" },
        ];

        TASK_FIELDS_TEMP.forEach((field, index) => {
            headingStyle(SHEET_2, 1, index + 1, "990000");
            const headerCell = SHEET_2.cell(1, index + 1);
            headerCell.value(field["lable"]);

        });

        PROJECT_TASKS.forEach((row, rowIndex) => {
            TASK_FIELDS_TEMP.forEach((field, columnIndex) => {
                const dataCell = SHEET_2.cell(rowIndex + 2, columnIndex + 1);
                if (field["name"] === "created_at" || field["name"] === "start_time" || field["name"] === "end_time") {
                    dataCell.value(row[field["name"]]?.toLocaleString("en-GB", { timeZone: "UTC" }));
                } else {
                    dataCell.value(row[field["name"]]?.toString());
                }
            });
        });

        for (let columnIndex = 1; columnIndex <= TASK_FIELDS_TEMP.length; columnIndex++) {
            SHEET_2.column(columnIndex).width(
                Math.max(15, ...PROJECT_TASKS.map(row => (row[TASK_FIELDS_TEMP[columnIndex - 1].name]?.toString() || '').length))
            );
        }

        for (let rowIndex = 1; rowIndex <= PROJECT_TASKS.length + 1; rowIndex++) {
            const isHeaderRow = rowIndex === 1;
            const defaultHeight = isHeaderRow ? 35 : 15;

            const maxHeight = Math.max(
                ...TASK_FIELDS_TEMP.map((field, columnIndex) => {
                    const cellValue = SHEET_2.cell(rowIndex, columnIndex + 1).value();
                    return cellValue ? (cellValue.toString().split('\n').length * defaultHeight) : defaultHeight;
                })
            );

            SHEET_2.row(rowIndex).height(maxHeight);
        }


        //=========== File 1 Sheet 3 ===========

        const SHEET_3 = await workbook.addSheet("Receipt");

        const RECEIPT = OVERALL_DETAILS["receipts"];

        const RECEIPT_FIELDS = [
            { name: "project_id", lable: "Project ID" },
            { name: "client", lable: "Client" },
            { name: "created_by", lable: "Worker" },
            { name: "receipt_title", lable: "Receipt Title" },
            { name: "receipt_city", lable: "City" },
            { name: "receipt_type", lable: "Receipt Type" },
            { name: "receipt_date", lable: "Receipt Date" },
            { name: "details", lable: "Description" },
            { name: "receipt_value", lable: "Value" },
        ];
        RECEIPT_FIELDS.forEach((field, index) => {
            headingStyle(SHEET_3, 1, index + 1, "990000");
            SHEET_3.cell(1, index + 1).value(field["lable"]);
        });
        RECEIPT.forEach((row, rowIndex) => {
            RECEIPT_FIELDS.forEach((field, columnIndex) => {
                if (field["name"] === "created_at") {
                    SHEET_3.cell(rowIndex + 2, columnIndex + 1).value(
                        row[field["name"]]?.toLocaleString("en-GB", { timeZone: "UTC" })
                    );
                } else if (field["name"] === "receipt_date") {
                    SHEET_3.cell(rowIndex + 2, columnIndex + 1).value(
                        row[field["name"]]?.toLocaleString("en-GB", { timeZone: "UTC" })
                    );
                } else if (field["name"] === "receipt_value") {
                    SHEET_3.cell(rowIndex + 2, columnIndex + 1).value(
                        parseFloat(row[field["name"]])
                    );
                } else {
                    SHEET_3.cell(rowIndex + 2, columnIndex + 1).value(
                        row[field["name"]]?.toString()
                    );
                }
            });
        });

        for (let columnIndex = 1; columnIndex <= RECEIPT_FIELDS.length; columnIndex++) {
            SHEET_3.column(columnIndex).width(
                Math.max(15, ...RECEIPT.map(row => (row[RECEIPT_FIELDS[columnIndex - 1].name]?.toString() || '').length))
            );
        }


        const buffer = await workbook.outputAsync();
        const upload = await uploadFileToAWS(buffer);
        const fileUrl = upload?.fileUrl;
        const fileName = upload.fileName;
        return { fileUrl, fileName: fileName };

        // return { fileUrl: "", fileName: "fileName" };

    } catch (error) { }
}
