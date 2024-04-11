import { GraphQLError } from "graphql";
import { prisma } from "../db";
import { QUERY, empTrackerLogs } from "../log";
import { headingStyle, uploadFileToAWS } from "./reports.resolver";
import { getProjectDataByDate } from "./project.resolver";
import { getNotInBudgetExpences } from "./expence.resolver";
import { calculateHourlyOvertimePercentage, countOvertimeOccurrences, correctHeadAndTail} from "./calculation-helpers.resolver";
import { createAWS_S3_URL } from "./worker_report.resolver";
const XlsxPopulate = require("xlsx-populate");

export function precisionRound(number, precision) {
    const factor = Math.pow(10, 3);
    return Math.round(number * factor) / factor;
}

export function calculatePercentage(value, percentage) {
    return (value / 100) * percentage;
}

export async function getCostMarginBudgetDashboardDetails(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} getCostMarginBudgetDashboardDetails`);
    let PROJECT_ID = args.input.project_id;
    if (PROJECT_ID === null) {
        PROJECT_ID = "test";
    }
    try {
        const projects = await prisma.project.findUnique({
            where: { id: PROJECT_ID },
            select: {
                client: { select: { name: true } },
                project_area: true,
                name: true,
                budget: true,
                end_date: true,
                start_date: true,
                year: true,
                margin: true,
                fee: true,
                contract_value: true,
                project_manager: {
                    select: { user: { select: { first_name: true, last_name: true } } },
                },
                area_manager: {
                    select: { user: { select: { first_name: true, last_name: true } } },
                },
                ProjectActivities: {
                    include: {
                        project_activity_id: {
                            include: {
                                project_worker_id: {
                                    select: {
                                        user: { select: { first_name: true, last_name: true } },
                                        id: true,
                                        worker_type: true,
                                        hourly_wages: true,
                                    },
                                },
                                tasks: true,
                            },
                        },
                    },
                },
            },
        });

        const budget: any = projects?.budget;
        const CONTRACT_VALUE = projects?.contract_value;
        let margin: any = projects?.margin;
        let fee: any = projects?.fee;
        margin = calculatePercentage(CONTRACT_VALUE, margin);
        fee = calculatePercentage(CONTRACT_VALUE, fee);

        const projectWorkers = {};
        const workers = [];
        projects?.ProjectActivities?.forEach((projectActivities) => {
            projectActivities?.project_activity_id?.forEach((projectWorker) => {
                const hourly_wages: any = projectWorker?.project_worker_id?.hourly_wages;
                const clientName = projects?.client?.name;
                const min_wages = hourly_wages / 60;
                let totalTime = 0;
                let totalCost = 0;
                if (projectWorker?.tasks?.length > 0) {

                    projectWorker?.tasks?.forEach((taskDetails) => {
                        if (taskDetails.deleted_at === null) {
                            const startDate: any = new Date(taskDetails?.start_time);
                            const endDate: any = new Date(taskDetails?.end_time);
                            const timeDiffInSeconds = Math.floor((endDate - startDate) / 1000);
                            const minutes = Math.floor(timeDiffInSeconds / 60);
                            const overtimeMap = calculateHourlyOvertimePercentage(startDate, endDate, projectWorker?.project_worker_id?.worker_type === "EMPLOYEE");
                            let tasksOvertime = countOvertimeOccurrences(overtimeMap);
                            tasksOvertime = correctHeadAndTail(startDate, endDate, overtimeMap, tasksOvertime);

                            const taskDurationHours = tasksOvertime[0];
                            let TOTAL_COST = 0;

                            if (clientName !== "Ferie / Permesso - FraminiaECS") {
                                TOTAL_COST = hourly_wages * taskDurationHours;
                                for (const overtimePercent in tasksOvertime) {
                                    if (overtimePercent !== "0" && tasksOvertime[overtimePercent]) {
                                        TOTAL_COST += (hourly_wages * tasksOvertime[overtimePercent]) * (1 + (parseInt(overtimePercent) / 100));
                                    }
                                }
                            }
                            totalTime += minutes;
                            totalCost += TOTAL_COST;
                        }
                    });
                    const workerObj = {
                        id: projectWorker?.project_worker_id?.id,
                        name: `${projectWorker?.project_worker_id?.user?.first_name} ${projectWorker?.project_worker_id?.user?.last_name}`,
                        totalCost: precisionRound(totalCost, 2)
                    };
                    if (projectWorkers[projectWorker?.project_worker_id?.id]) {
                        const tCost = projectWorkers[projectWorker?.project_worker_id?.id].totalCost + totalCost;
                        projectWorkers[projectWorker?.project_worker_id?.id].totalCost = precisionRound(tCost, 2);
                    } else {
                        projectWorkers[projectWorker?.project_worker_id?.id] = workerObj;

                    }
                }
            });
        });
        for (const key in projectWorkers) {
            workers.push(projectWorkers[key]);
        }
        let totalCostOfWorker = 0;
        for (const key in projectWorkers) {
            totalCostOfWorker += projectWorkers[key]["totalCost"];
        }
        const data = {
            name: projects?.name,
            project_manager: projects?.project_manager,
            area_manager: projects?.area_manager,
            budget: parseFloat(budget),
            margin: precisionRound(margin, 2),
            fee: precisionRound(fee, 2),
            totalCostOfWorker: precisionRound(totalCostOfWorker, 2),
        };

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


        const receipt = rawdata.map((attachment) => {
            const KEY = `${process.env.S3_BUCKET_FOLDER_NAME}/${encodeURIComponent(attachment.file_name)}`;
            const fileUrl = createAWS_S3_URL(KEY);
            return {
                ...attachment,
                fileUrl,
            };
        });
        if (!receipt) {
            return new GraphQLError("Attachments not Found.", {
                extensions: {
                    code: "SOMETHING_BAD_HAPPENED",
                },
            });
        }
        let totalValueOfReceipt: any = 0;
        for (const key in receipt) {
            const val: any = receipt[key]["receipt_value"];
            totalValueOfReceipt += parseFloat(val);
        }
        data["totalValueOfReceipt"] = precisionRound(totalValueOfReceipt, 3);

        const rawdataofExpense = await prisma.expense.findMany({
            where: {
                project_id: PROJECT_ID, AND: { reimbursed: { equals: true }, AND: { deleted_at: { equals: null } } }
            }
        });
        const expences = rawdataofExpense.map((attachment) => {
            const KEY = `${process.env.S3_BUCKET_FOLDER_NAME}/${encodeURIComponent(attachment.file_name)}`;
            const fileUrl = createAWS_S3_URL(KEY);
            return {
                ...attachment,
                fileUrl
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
            const val: any = expences[key]["cost"];
            totalCostOfExpences += parseFloat(val);
        }
        data["totalCostOfExpences"] = precisionRound(totalCostOfExpences, 2);

        data["cost"] = data["totalValueOfReceipt"] + data["totalCostOfWorker"];


        return data;


    } catch (error) {
        console.log(error, "error from getCostMarginBudgetDetails");
    }

}

export async function generateDashboard_1_CSV(parent, args, contextValue) {
    empTrackerLogs(`${QUERY} generateDashboard_1_CSV`);
    try {

        const OVERALL_DETAILS_ARGS = {
            input: {
                project_id: args.input.project_id
            }
        };

        const OVERALL_DETAILS: any = await getCostMarginBudgetDashboardDetails("", OVERALL_DETAILS_ARGS, "");
        const OVERALL_DETAILS_ARRAY = [];
        for (const iterator in OVERALL_DETAILS) {
            if (iterator == "project_manager" || iterator == "area_manager") {
                OVERALL_DETAILS_ARRAY.push({ "Name": iterator.toUpperCase(), "Value": `${OVERALL_DETAILS[iterator]["user"]["first_name"]} ${OVERALL_DETAILS[iterator]["user"]["last_name"]} ` });
            } else if (iterator == "totalCostOfWorker") {
                OVERALL_DETAILS_ARRAY.push({ "Name": "TOTAL COST OF WORKER", "Value": OVERALL_DETAILS[iterator] });
            }
            else if (iterator == "totalValueOfReceipt") {
                OVERALL_DETAILS_ARRAY.push({ "Name": "TOTAL VALUE OF RECEIPT", "Value": OVERALL_DETAILS[iterator] });
            }
            else if (iterator == "totalCostOfExpences") {
                OVERALL_DETAILS_ARRAY.push({ "Name": "TOTAL COST OF EXPENCES", "Value": OVERALL_DETAILS[iterator] });
            }
            else {
                OVERALL_DETAILS_ARRAY.push({ "Name": iterator.toUpperCase(), "Value": OVERALL_DETAILS[iterator] });
            }
        }
        const workbook = await XlsxPopulate.fromBlankAsync();
        //=========== File 1 Sheet 1 ===========
        const SHEET_0 = await workbook.sheet(0).name("Project Details");
        const SHEET_0_FIELDS = [
            "Name", "Value"];
        SHEET_0_FIELDS.forEach((field, index) => {
            headingStyle(SHEET_0, 1, index + 1, "990000");
            SHEET_0.cell(1, index + 1).value(field);
        });
        OVERALL_DETAILS_ARRAY.forEach((row, rowIndex) => {
            SHEET_0_FIELDS.forEach((field, columnIndex) => {
                SHEET_0.cell(rowIndex + 2, columnIndex + 1).value(row[field]);
            });
        });

        // ================================================================================ 
        // ================================================================================ 
        const GET_PROJECT_DATA_BY_DATE_ARGS = {
            input: {
                project_id: args.input.project_id,
                toDate: args.input.box_one_toDate,
                fromDate: args.input.box_one_fromDate
            }
        };
        const GET_PROJECT_DATA_BY_DATE = await getProjectDataByDate("", GET_PROJECT_DATA_BY_DATE_ARGS, "");


        //=========== File 1 Sheet 1 ===========
        const SHEET_1 = await workbook.addSheet("Workers");
        const SHEET_1_FIELDS = ["name", "totalCost"];
        SHEET_1_FIELDS.forEach((field, index) => {
            headingStyle(SHEET_1, 1, index + 1, "990000");
            SHEET_1.cell(1, index + 1).value(field);
        });
        GET_PROJECT_DATA_BY_DATE["workers"].forEach((row, rowIndex) => {
            SHEET_1_FIELDS.forEach((field, columnIndex) => {
                SHEET_1.cell(rowIndex + 2, columnIndex + 1).value(row[field]?.toString());
            });
        });
        //=========== File 1 Sheet 2 ===========
        const SHEET_2 = await workbook.addSheet("Receipt");
        const SHEET_2_FIELDS = ["receipt_title", "receipt_value", "receipt_date"];
        SHEET_2_FIELDS.forEach((field, index) => {
            headingStyle(SHEET_2, 1, index + 1, "990000");
            SHEET_2.cell(1, index + 1).value(field);
        });
        GET_PROJECT_DATA_BY_DATE["receipts"].forEach((row, rowIndex) => {
            SHEET_2_FIELDS.forEach((field, columnIndex) => {
                if (field === "receipt_date") {
                    SHEET_2.cell(rowIndex + 2, columnIndex + 1).value(row[field]?.toLocaleString("en-GB", { timeZone: "UTC" }));
                } else {
                    SHEET_2.cell(rowIndex + 2, columnIndex + 1).value(row[field]?.toString());
                }
            });
        });

        //=========== File 1 Sheet 3 ===========
        const SHEET_3 = workbook.addSheet("expences");
        const SHEET_3_FIELDS = ["title", "cost"];
        SHEET_3_FIELDS.forEach((field, index) => {
            headingStyle(SHEET_3, 1, index + 1, "990000");
            SHEET_3.cell(1, index + 1).value(field);
        });
        GET_PROJECT_DATA_BY_DATE["expences"].forEach((row, rowIndex) => {
            SHEET_3_FIELDS.forEach((field, columnIndex) => {
                SHEET_3.cell(rowIndex + 2, columnIndex + 1).value(row[field]?.toString());
            });
        });

        // ================================================================================ 
        // ================================================================================ 

        const GET_NOT_IN_BUDGET_EXPENCES_ARG = {
            input: {
                project_id: args.input.project_id,
                toDate: args.input.box_two_toDate,
                fromDate: args.input.box_two_fromDate
            }
        };
        const GET_NOT_IN_BUDGET_EXPENCES = await getNotInBudgetExpences("", GET_NOT_IN_BUDGET_EXPENCES_ARG, "");

        //=========== File 1 Sheet 4 ===========   
        const SHEET_4 = workbook.addSheet("Expences Not in Budget");
        const SHEET_4_FIELDS = ["title", "cost", "expense_date"];
        SHEET_4_FIELDS.forEach((field, index) => {
            headingStyle(SHEET_4, 1, index + 1, "990000");
            SHEET_4.cell(1, index + 1).value(field);
        });
        GET_NOT_IN_BUDGET_EXPENCES["expences"].forEach((row, rowIndex) => {
            SHEET_4_FIELDS.forEach((field, columnIndex) => {
                if (field === "expense_date") {
                    SHEET_4.cell(rowIndex + 2, columnIndex + 1).value(row[field]?.toLocaleString("en-GB", { timeZone: "UTC" }));
                } else {
                    SHEET_4.cell(rowIndex + 2, columnIndex + 1).value(row[field]?.toString());
                }
            });
        });

        const buffer = await workbook.outputAsync();
        const upload = await uploadFileToAWS(buffer);
        const fileUrl = upload?.fileUrl;
        const fileName = upload.fileName;
        return { fileUrl, fileName: fileName };

    } catch (err) {
        console.log(err, "Error in dashboard_1_CSV");
    }
}