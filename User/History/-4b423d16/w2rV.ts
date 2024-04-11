import { MUTATION, QUERY, empTrackerLogs } from "../log";
import { prisma } from "../db";
import { GraphQLError } from "graphql";
import moment, { duration } from "moment";
import { precisionRound } from "./dashboard_reports.resolver";
import { HOLIDAYS_LIST, HOLIDAY_DATE_FORMAT } from "../utils/holidays";
import { DateTime } from 'luxon';



export async function getActivityTasksOfWorker(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} getActivityTasksOfWorker`);
    const tasks = await prisma.activityWorkerTasks.findMany({
        orderBy: { start_time: 'desc' },
        where: {
            deleted_at: { equals: null },
            activity_worker: {
                activity_id: args.input.activity_id,
                AND: {
                    worker_id: contextValue.profile_id,
                },
            },
        },
        include: {
            activity_worker: {
                select: {
                    project_worker_id: {
                        select: { user: { select: { first_name: true, last_name: true } } },
                    },
                    project_activity_id: {
                        select: {
                            project_id: { select: { client: { select: { name: true } } } },
                        },
                    },
                },
            },
        },
    });
    tasks.forEach((task) => {
        const CREATED_BY = `${task?.activity_worker?.project_worker_id?.user?.first_name} ${task?.activity_worker?.project_worker_id?.user?.last_name}`;
        task["created_by"] = CREATED_BY;
        const startDate: any = new Date(task?.start_time);
        const endDate: any = new Date(task?.end_time);
        const timeDiffInSeconds = Math.floor((endDate - startDate) / 1000);
        const minutes = Math.floor(timeDiffInSeconds / 60);
        const totalMinutes = minutes % 60;
        const hours = Math.floor(minutes / 60);
        const clientName =
            task?.activity_worker?.project_activity_id?.project_id?.client?.name;
        task["client_name"] = clientName;
        task["total_hrs"] = `${hours} Hours ${totalMinutes > 0 ? ` ${totalMinutes} Minutes` : ""
            }`;
    });
    return { tasks };
}

export async function getTasksByID(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} getTasksByID`);
    const TASK_ID = args?.input?.task_id;
    const GET_TASK = await prisma.activityWorkerTasks.findUnique({
        where: { id: TASK_ID },
        include: {
            activity_worker: {
                select: {
                    project_worker_id: {
                        select: { user: { select: { first_name: true, last_name: true } } },
                    },
                },
            },
        },
    });
    const WORKER_NAME = `${GET_TASK?.activity_worker?.project_worker_id?.user?.first_name} ${GET_TASK?.activity_worker?.project_worker_id?.user?.last_name}`;
    GET_TASK["worker_name"] = WORKER_NAME;
    return GET_TASK;
}

const convertToItalyTimezone = (date: Date): Date => {
    const offset = 60; // Italy's timezone offset from UTC in minutes (currently, it's UTC+1)
    const newDate = new Date(date.getTime() + offset * 60000);
    return newDate;
};

export async function createActivityTask(parent, args, contextValue) {
    empTrackerLogs(`${MUTATION} createActivityTask`);
    let { activity_id, name, start_time, end_time, ferie } = args.input;
    if (!activity_id) {
        return new GraphQLError("Activity Id Not Provided.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }

    if (!name) {
        return new GraphQLError("Name Not Provided.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }

    if (!start_time) {
        return new GraphQLError("start_time Not Provided.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }

    if (!end_time) {
        return new GraphQLError("end_time Not Provided.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }
    const workerActivityObj = await prisma.activityWorkers.findFirst({
        where: { activity_id: activity_id, worker_id: contextValue.profile_id },
    });

    const data = await prisma.activityWorkerTasks.create({
        data: {
            activity_worker: { connect: { id: workerActivityObj.id } },
            name,
            start_time,
            end_time,
            created_at: new Date(),
            ferie: ferie
        },
    });

    return { success: true };
}

export async function deleteActivityTask(parent, args, contextValue) {
    empTrackerLogs(`${MUTATION} deleteActivityTask`);
    const TASK_ID = args.input.task_id;
    if (!TASK_ID) {
        return new GraphQLError("TASK Id Not Provided.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }
    const task = await prisma.activityWorkerTasks.findUnique({
        where: { id: TASK_ID },
    });
    if (!task || task.deleted_at != null) {
        throw new GraphQLError("task Not Found", {
            extensions: {
                code: "BAD_USER_INPUT",
            },
        });
    }
    const deleteTask = await prisma.activityWorkerTasks.update({
        where: { id: TASK_ID },
        data: { deleted_at: new Date() },
    });
    return {
        status: "SUCCESS",
        message: "Task has been Deleted successfully",
    };
}

export async function updateTask(parent, args, contextValue) {
    const TASK_ID = args.input.id;
    const GET_TASK = await prisma.activityWorkerTasks.findUnique({
        where: { id: TASK_ID },
    });

    const DATA = { ...GET_TASK, ...args.input };

    if (DATA?.ferie === "false") {
        DATA["ferie"] = false
    }
    if (DATA?.ferie === "true") {
        DATA["ferie"] = true
    }

    const UPDATE_TASK = await prisma.activityWorkerTasks.update({
        where: { id: TASK_ID },
        data: DATA,
    });
    return {
        status: "SUCCESS",
        message: "Task has been Updated successfully",
    };
}

export function calculateTimeOfTask(startTime, endTime) {
    const taskMoment = moment(startTime);
    const taskEndMoment = moment(endTime);
    const taskDuration = duration(taskEndMoment.diff(taskMoment));
    const taskDurationMin = taskDuration.asMinutes();

    return taskDurationMin;
}



function isHoliday(date) {
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();

    const holidays = [
        `01-01-${year}`,  // New Year's Day
        `01-06-${year}`,  // Epiphany
        `04-25-${year}`,  // Liberation Day
        `05-01-${year}`,  // Labour Day
        `06-02-${year}`,  // Republic Day
        `08-15-${year}`,  // Assumption Day
        `11-01-${year}`,  // All Saints' Day
        `12-08-${year}`,  // Immaculate Conception
        `12-25-${year}`,  // Christmas Day
        `12-26-${year}`   // St. Stephen's Day
    ];

    const isSunday = date.getDay() === 0;
    return isSunday || holidays.includes(`${month}-${day}-${year}`);
}
