import { GraphQLError } from "graphql";
import { prisma } from "../db";
import { MUTATION, QUERY, empTrackerLogs } from "../log";
import { precisionRound } from "./dashboard_reports.resolver";
import { correctHeadAndTail, createAWS_S3_URL } from "./worker_report.resolver";
import { calculateTimeOfTask } from "./tasks.resolver";
import { countOvertimeOccurrences, calculateHourlyOvertimePercentage } from "./worker_report.resolver";

export async function createProject(parent, args, contextValue) {
    empTrackerLogs(`${MUTATION} createProject`);
    /*  if (
         !args?.input?.state_code ||
         args?.input?.state_code?.trim()?.length === 0
     ) {
         return new GraphQLError("state_code Not Provided.", {
             extensions: {
                 code: "SOMETHING_BAD_HAPPENED",
             },
         });
     }
     if (!args?.input?.offer_id || args?.input?.offer_id?.trim()?.length === 0) {
         return new GraphQLError("offer_id Not Provided.", {
             extensions: {
                 code: "SOMETHING_BAD_HAPPENED",
             },
         });
     }
     if (
         !args?.input?.project_id ||
         args?.input?.project_id?.trim()?.length === 0
     ) {
         return new GraphQLError("project_id Not Provided.", {
             extensions: {
                 code: "SOMETHING_BAD_HAPPENED",
             },
         });
     }
     if (
         !args?.input?.client_note ||
         args?.input?.client_note?.trim()?.length === 0
     ) {
         return new GraphQLError("client_note Not Provided.", {
             extensions: {
                 code: "SOMETHING_BAD_HAPPENED",
             },
         });
     }
     if (!args?.input?.budget) {
         return new GraphQLError("budget Not Provided.", {
             extensions: {
                 code: "SOMETHING_BAD_HAPPENED",
             },
         });
     }
     if (!args?.input?.start_date) {
         return new GraphQLError("start_date Not Provided.", {
             extensions: {
                 code: "SOMETHING_BAD_HAPPENED",
             },
         });
     }
     if (!args?.input?.end_date) {
         return new GraphQLError("end_date Not Provided.", {
             extensions: {
                 code: "SOMETHING_BAD_HAPPENED",
             },
         });
     } */

    if (args?.input?.margin > 100 || args?.input?.margin < 0) {
        return new GraphQLError("margin should in between 0 to 100.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }

    if (args?.input?.fee > 100 || args?.input?.fee < 0) {
        return new GraphQLError("fee should in between 0 to 100.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }

    if (args.input.project_id) {
        const getProjectId = await prisma.project.findMany({
            where: {
                project_id: args?.input?.project_id?.trim(),
                AND: { deleted_at: { equals: null } },
            },
        });
        if (getProjectId.length > 0) {
            return new GraphQLError("Project Id already exists", {
                extensions: {
                    code: "SOMETHING_BAD_HAPPENED",
                },
            });
        }
    }

    const projectData = {
        name: args?.input?.name?.trim(),
        address: args?.input?.address || null,
        description: args?.input?.description || null,
        year: new Date().getFullYear() || null,
        area_id: args?.input?.area_id || null,
        state_code: args?.input?.state_code || null,
        offer_id: args?.input?.offer_id || null,
        invoice_id: args?.input?.invoice_id || null,
        client_id: args?.input?.client_id || null,
        project_id: args?.input?.project_id || null,
        area_manager_id: args?.input?.area_manager_id || null,
        project_manager_id: args?.input?.project_manager_id || null,
        created_by: contextValue?.profile_id || null,
        billing_time: args?.input?.billing_time || null,
        invoice_details: args?.input?.invoice_details || null,
        check: args?.input?.check || null,
        budget: args?.input?.budget || null,
        client_note: args?.input?.client_note || null,
        note: args?.input?.note || null,
        event_date: args?.input?.event_date || null,
        start_date: args?.input?.start_date || null,
        end_date: args?.input?.end_date || null,
        deleted_at: null,
        latitude: args?.input?.latitude || null,
        longitude: args?.input?.longitude || null,
        project_reference: args?.input?.project_reference || null,
        payment_check: args?.input?.payment_check || false,
        contract_value: args?.input?.contract_value || null,
        margin: args?.input?.margin || null,
        fee: args?.input?.fee || null,
        client_reference: args?.input?.client_reference || null,
        reporting_file: args?.input?.reporting_file || false,
        check_planning: args?.input?.check_planning || false,
        status_code: args?.input?.status_code || null,
        quality_plan: args?.input?.quality_plan || false,
        expeses_to_be_reimbursed: args?.input?.expeses_to_be_reimbursed || false,
        expeses_included: args?.input?.expeses_included || false,
        status: args?.input?.status || null,
        client_group: args?.input?.client_group || null,
        client_manager: args?.input?.client_manager || null,
        sector_id: args?.input?.sector_id || null,
        note_compilatore: args?.input?.note_compilatore || null,
        note_check_pm: args?.input?.note_check_pm || null,
        note_resp_area: args?.input?.note_resp_area || null,
    };
    const project = await prisma.project.create({
        data: projectData,
    });
    return {
        status: "SUCCESS",
        message: `Project has been successfully created With ID ${project["project_id"]}`,
    };
}

export async function findAllProject(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} findAllProject`);
    const data = await prisma.project.findMany({
        where: {
            client_id: args.input.client_id,
            AND: {
                deleted_at: { equals: null },
            },
        },
        include: {
            client: true,
            ProjectActivities: {
                where: { deleted_at: { equals: null } },
                select: { project_activity_id: true },
            },
        },
    });



    data.forEach((project) => {
        project["acivities_count"] = project.ProjectActivities?.length;
        project["workers_count"] = 0;
    });

    data.forEach((project) => {
        const workerObj = {};
        project.ProjectActivities?.forEach((activities) => {
            activities.project_activity_id.forEach((worker) => {
                if (worker.deleted_at === null) {
                    if (!workerObj[worker.worker_id]) {
                        workerObj[worker.worker_id] = 1;
                        project["workers_count"] += 1;
                    }
                }
            });
        });
    });
    return data;
}

export async function allProjectList(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} allProjectList`);
    const data = await prisma.project.findMany({
        where: {
            deleted_at: { equals: null }, AND: { client: { deleted_at: { equals: null } } }
        },
        select: { id: true, name: true, project_id: true, deleted_at: true },
    });

    return data;
}

export async function getProjectById(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} getProjectById`);
    const data = await prisma.project.findUnique({
        where: { id: args.input.id },
        select: {
            id: true,
            name: true,
            project_id: true,
            description: true,
            budget: true,
            year: true,
            address: true,
            state_code: true,
            offer_id: true,
            invoice_details: true,
            check: true,
            billing_time: true,
            client_note: true,
            invoice_id: true,
            start_date: true,
            end_date: true,
            event_date: true,
            note_compilatore: true,
            note_check_pm: true,
            note_resp_area: true,
            project_area: { select: { id: true, name: true } },
            area_manager: {
                select: {
                    id: true,
                    user: { select: { first_name: true, last_name: true } },
                },
            },
            project_created_by: {
                select: {
                    id: true,
                    user: { select: { first_name: true, last_name: true } },
                },
            },
            project_manager: {
                select: {
                    id: true,
                    user: { select: { first_name: true, last_name: true } },
                },
            },

            deleted_at: true,
            client_id: true,
            note: true,
            latitude: true,
            longitude: true,
            margin: true,
            fee: true,
            check_planning: true,
            client_reference: true,
            contract_value: true,
            expeses_included: true,
            expeses_to_be_reimbursed: true,
            payment_check: true,
            project_reference: true,
            reporting_file: true,
            quality_plan: true,
            status_code: true,
            status: true,
            client_group: true,
            client_manager: true,
            project_sector: { select: { id: true, name: true } }
        },
    });

    if (!data || data.deleted_at != null) {
        return new GraphQLError("Project Not Found.", {
            extensions: {
                code: "CONFLICT",
            },
        });
    }
    return data;
}

export async function deleteProject(parent, args, contextValue) {
    empTrackerLogs(`${MUTATION} deleteProject`);
    const getProject = await prisma.project.findUnique({
        where: { id: args.input.id },
    });

    if (!getProject || getProject.deleted_at != null) {
        return new GraphQLError("Project has been Already Deleted.", {
            extensions: {
                code: "CONFLICT",
            },
        });
    }
    const project = await prisma.project.update({
        where: { id: args.input.id },
        data: { deleted_at: new Date() },
    });

    if (project) {
        return {
            status: "SUCCESS",
            message: "Project has been Deleted successfully",
        };
    }
}

export async function updateProject(parent, args, contextValue) {
    empTrackerLogs(`${MUTATION} updateProject`);
    if (!args.input.id) {
        return new GraphQLError("Project Id Not Provided.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }

    const getProject = await prisma.project.findUnique({
        where: { id: args.input.id },
    });

    if (!getProject || getProject.deleted_at != null) {
        return new GraphQLError("Project has been Already Deleted.", {
            extensions: {
                code: "CONFLICT",
            },
        });
    }

    if (args.input.year) {
        args.input.year = parseInt(args?.input?.year.slice(0, 4));
    }
    const UPDATED_DATA = {};

    for (const field in args.input) {
        if (args.input[field]) {
            UPDATED_DATA[field] = args.input[field];
        }
    }

    const project = await prisma.project.update({
        where: { id: args.input.id },
        data: UPDATED_DATA,
    });
    return {
        status: "SUCCESS",
        message: "Project has been Updated successfully",
    };
}

export async function getProjectDataByDate(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} getProjectDataByDate`);
    const FROM_DATE = args.input.fromDate;
    let TO_DATE = args.input.toDate;
    const PROJECT_ID = args.input.project_id;
    if (!TO_DATE) {
        const NewEndDate = new Date(FROM_DATE);
        NewEndDate.setMinutes(1439);
        NewEndDate.setSeconds(59);
        TO_DATE = NewEndDate;
    }
    const result = {};
    let project;

    /* ==============W=O=R=K=E=R=S================= */
    /* ==============W=O=R=K=E=R=S================= */
    if (!FROM_DATE) {
        const projects = await prisma.activityWorkers.findMany({
            where: {
                project_activity_id: {
                    projectId: { equals: PROJECT_ID },

                },
                AND: {
                    deleted_at: { equals: null }
                },
            }, include: {
                project_worker_id: {
                    select: {
                        user: { select: { first_name: true, last_name: true } },
                        id: true,
                        hourly_wages: true,
                        worker_type: true,
                    },
                },
                tasks: { where: { deleted_at: { equals: null } } },
            }
        });
        project = projects;
    } else {
        const projects = await prisma.activityWorkers.findMany({
            where: {
                project_activity_id: {
                    projectId: { equals: PROJECT_ID },
                },
                AND: {
                    created_at: { gte: FROM_DATE }, AND: { created_at: { lte: TO_DATE } }
                },
            }, include: {
                project_worker_id: {
                    select: {
                        user: { select: { first_name: true, last_name: true } },
                        id: true,
                        worker_type: true,
                        hourly_wages: true,
                    },
                },
                tasks: { where: { deleted_at: { equals: null } } },
            }
        });
        project = projects;
    }
    const projectWorkers = {};
    const workers = [];
    project?.forEach((projectWorker) => {
        const hourly_wages: any = projectWorker?.project_worker_id?.hourly_wages;
        const min_wages = hourly_wages / 60;
        let totalTime = 0;
        let totalCost = 0;
        let ordinary_hours = 0
        let overtime_hours = 0;
        if (projectWorker?.tasks?.length > 0) {
            projectWorker?.tasks?.forEach((taskDetails) => {
                const startDate: any = new Date(taskDetails?.start_time);
                const endDate: any = new Date(taskDetails?.end_time);
                const timeDiffInSeconds = Math.floor((endDate - startDate) / 1000);
                const minutes = Math.floor(timeDiffInSeconds / 60);
                const overtimeMap = calculateHourlyOvertimePercentage(taskDetails?.start_time, taskDetails?.end_time, projectWorker?.project_worker_id?.worker_type === "EMPLOYEE");
                let tasksOvertime = countOvertimeOccurrences(overtimeMap);
                tasksOvertime = correctHeadAndTail(task?.start_time, task?.end_time, overtimeMap, tasksOvertime);
                const taskDurationHours = tasksOvertime[0];
                ordinary_hours += taskDurationHours
                overtime_hours += overtimeMap.size - taskDurationHours
                let TOTAL_COST = hourly_wages * taskDurationHours;
                for (const overtimePercent in tasksOvertime) {
                    if (overtimePercent !== "0" && tasksOvertime[overtimePercent]) {
                        TOTAL_COST += (hourly_wages * tasksOvertime[overtimePercent]) * (1 + (parseInt(overtimePercent) / 100));
                    }
                }
                totalTime += minutes;
                totalCost += TOTAL_COST;
            });
            const workerObj = {
                id: projectWorker?.project_worker_id?.id,
                name: `${projectWorker?.project_worker_id?.user?.first_name} ${projectWorker?.project_worker_id?.user?.last_name}`,
                totalCost: precisionRound(totalCost, 2),
                totalTime: precisionRound((totalTime / 60), 2),
                ordinary_hours: ordinary_hours,
                overtime_hours: overtime_hours
            };
            if (projectWorkers[projectWorker?.project_worker_id?.id]) {
                const tCost = projectWorkers[projectWorker?.project_worker_id?.id].totalCost + totalCost;
                projectWorkers[projectWorker?.project_worker_id?.id].totalCost = precisionRound(tCost, 2);
                projectWorkers[projectWorker?.project_worker_id?.id].totalTime = precisionRound((totalTime / 60), 2);
                projectWorkers[projectWorker?.project_worker_id?.id].ordinary_hours += ordinary_hours
                projectWorkers[projectWorker?.project_worker_id?.id].overtime_hours += overtime_hours
            } else {
                projectWorkers[projectWorker?.project_worker_id?.id] = workerObj;
            }
        }
    });
    for (const key in projectWorkers) {
        workers.push(projectWorkers[key]);
    }
    let totalCostOfWorker = 0;
    for (const key in projectWorkers) {
        totalCostOfWorker += projectWorkers[key]["totalCost"];
    }
    result["workers"] = workers;
    result["totalCostOfWorker"] = precisionRound(totalCostOfWorker, 3);

    /* ==============R=E=C=E=I=P=T================= */
    /* ==============R=E=C=E=I=P=T================= */

    let receiptRawData;
    if (!FROM_DATE) {
        const rawdata = await prisma.activityAttachments.findMany({
            where: {
                type: "RECEIPT",
                AND: {
                    deleted_at: { equals: null }, AND: {
                        activity_attachment_id: { projectId: { equals: PROJECT_ID } }
                    }
                },

            },
        });
        receiptRawData = rawdata;
    } else {
        const rawdata = await prisma.activityAttachments.findMany({
            where: {
                type: "RECEIPT",
                AND: {
                    deleted_at: { equals: null }, AND: {
                        activity_attachment_id: { projectId: { equals: PROJECT_ID } },
                        AND: {
                            created_at: { gte: FROM_DATE }, AND: { created_at: { lte: TO_DATE } }
                        },
                    },
                }
            },
        });
        receiptRawData = rawdata;
    }
    const receipt = receiptRawData.map((attachment) => {
        const KEY = `${process.env.S3_BUCKET_FOLDER_NAME}/${encodeURIComponent(attachment.file_name)}`;
        let fileUrl = null;
        if (attachment.file_name) {
            fileUrl = createAWS_S3_URL(KEY);
        }
        return {
            ...attachment,
            fileUrl
        };
    });
    if (!receipt) {
        return new GraphQLError("Attachments not Found.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }
    let totalValueOfReceipt = 0;
    for (const key of receipt) {
        totalValueOfReceipt += parseFloat(key["receipt_value"]);
    }
    result["receipts"] = receipt;
    result["totalValueOfReceipt"] = precisionRound(totalValueOfReceipt, 3);


    /* ==============E=X=P=E=N=C=E================= */
    /* ==============E=X=P=E=N=C=E================= */

    let rawExpenceData;
    if (!FROM_DATE) {
        const rawdata = await prisma.expense.findMany({
            where: {
                deleted_at: { equals: null }, AND: {
                    project_id: PROJECT_ID, AND: { reimbursed: { equals: false } }
                }
            }, include: { expense_type: { select: { type: true } }, project_expence: { select: { project_id: true, client: { select: { name: true } } } } }
        });
        rawExpenceData = rawdata;
    } else {
        const rawdata = await prisma.expense.findMany({
            where: {
                deleted_at: { equals: null }, AND: {
                    project_id: PROJECT_ID, AND: {
                        created_at: { gte: FROM_DATE }, AND: { created_at: { lte: TO_DATE }, AND: { reimbursed: { equals: false } } }
                    },
                }
            }, include: { expense_type: { select: { type: true } }, project_expence: { select: { project_id: true, client: { select: { name: true } } } } }
        });
        rawExpenceData = rawdata;
    }


    const expences = rawExpenceData.map((attachment) => {
        const KEY = `${process.env.S3_BUCKET_FOLDER_NAME}/${attachment.file_name}`;
        let fileUrl = null;
        if (attachment.file_name) {
            fileUrl = createAWS_S3_URL(KEY);
        }


        let expense_type = attachment['expense_type']['type']
        let client = attachment['project_expence']['client']['name']
        let project_id = attachment['project_expence']['project_id']

        return {
            ...attachment,
            fileUrl, expense_type, client, project_id
        };
    });
 
    if (!expences) {
        return new GraphQLError("Attachments not Found.", {
            extensions: {
                code: "SOMETHING_BAD_HAPPENED",
            },
        });
    }
    let totalCostOfExpences = 0;
    for (const key in expences) {
        totalCostOfExpences += parseFloat(expences[key]["cost"]);
    }
    result["expences"] = expences;
    result["totalCostOfExpences"] = precisionRound(totalCostOfExpences, 2);

    return result;
}

export async function getWorkersByProjectId(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} getWorkersByProjectId`);

    const getWorkers = await prisma.project.findFirst({
        where: { id: args.input.project_id },
        select: {
            ProjectActivities: {
                select: {
                    project_activity_id: {
                        select: {
                            project_worker_id: {
                                select: {
                                    user: { select: { first_name: true, last_name: true } },
                                    qualification: true,
                                    id: true,
                                },
                            },
                        },
                    },
                },
            },
        },
    });

    const rawWorkers = getWorkers.ProjectActivities;
    const workerGroup = {};
    const workers = [];
    rawWorkers.forEach((activity) => {
        activity.project_activity_id.forEach((worker) => {
            if (!workerGroup[worker.project_worker_id.id]) {
                workerGroup[worker.project_worker_id.id] = {
                    first_name: worker.project_worker_id.user.first_name,
                    last_name: worker.project_worker_id.user.last_name,
                    qualification: worker.project_worker_id.qualification,
                };
            }
        });
    });

    for (const key in workerGroup) {
        workers.push(workerGroup[key]);
    }

    return workers;
}
