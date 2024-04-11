<?php

namespace App\Services;

use App\Lib\Connectors\Serial;/*ok*/
use App\Lib\Connectors\Socket;
use App\Lib\Connectors\PrinterFile;
use App\Lib\POS\Ingenico\Device as POSDevice;
use App\Lib\POS\Ingenico\Message;
use App\Lib\POS\Ingenico\iSelf\Messages\Commands\StartOperation;
use App\Lib\POS\Ingenico\iSelf\Messages\Commands\SendingAmount;
use App\Lib\POS\Ingenico\iSelf\Messages\Responses\AsyncMessage;
use App\Lib\POS\Ingenico\iSelf\Messages\Commands\CancelTransaction;
use App\Lib\Printers\Device as PrinterDevice;
use App\Lib\Printers\KPM150H\Driver as DriverKPM150H;
use App\Lib\Printers\VKP80II\Driver as DriverVKP80II;
use App\Lib\Printers\KUBEII\Driver as DriverKUBEII;

use App\Device;

class TvmService
{
    protected $options;
    protected $bufferSize = 4096;
    protected $abort = false;
    protected $connector;
    protected $posResponseTimeout = 1000;

    public function __construct() {
        //$this->options = TicketCenter\Models\Devices::getOptionsArray();
        $this->bufferSize = ini_get('output_buffering');
    }

    public function getDriver($devices_code, $verbose = false) {
        $device = Device::active()->where('code',$devices_code)->first();
        switch ($device->code) {
            case 'POS_Ingenico_IUP250':
                $this->connector = new Serial(array(
                    'device' => $devices_code,
                    'baudRate' => $device->baudRate,
                    'parity' => $device->parity,
                    'bitChar' => $device->bitChar,
                    'stopBit' => $device->stopBit));
                try {
                    $driver = new POSDevice(array(
                        'connector' => $this->connector,
                        'terminalIdentifier' => $device->terminalIdentifier,
                        'debug' => $verbose
                    ));
                } catch (\Exception $e) {
                    $this->connector->close();
                    throw new \Exception($e->getMessage());
                }
                break;
            case 'POS_Ingenico_SELF5000':
                $this->connector = new Socket(array(
                    'address' => $device->address,
                    'port' => $device->port));
                try {
                    $driver = new POSDevice(array(
                        'connector' => $this->connector,
                        'terminalIdentifier' => $device->terminalIdentifier,
                        'debug' => $verbose
                    ));
                } catch (\Exception $e) {
                    $this->connector->close();
                    throw new \Exception($e->getMessage());
                }
                break;
            case 'POS_Ingenico_SELF5000_S':
                $this->connector = new Serial(array(
                    'device' => $devices_code,
                    'baudRate' => $device->baudRate,
                    'parity' => $device->parity,
                    'bitChar' => $device->bitChar,
                    'stopBit' => $device->stopBit));
                try {
                    $driver = new POSDevice(array(
                        'connector' => $this->connector,
                        'terminalIdentifier' => $device->terminalIdentifier,
                        'debug' => $verbose
                    ));
                } catch (\Exception $e) {
                    $this->connector->close();
                    throw new \Exception($e->getMessage());
                }
                break;
            case 'Custom_KPM150H':
                sleep(2);
                try {
                    $this->connector = new PrinterFile(array(
                        'directory' => $device->directory, 
                        'MDL' => $device->MDL, 
                        'debug' => $verbose));
                    $printer = new PrinterDevice(array(
                        'connector' => $this->connector, 
                        'debug' => $verbose));
                    $driver = new DriverKPM150H(array(
                        'device' => $printer));
                } catch (\Exception $e) {
                    if (!empty($this->connector)) {
                        $this->connector->close();
                    }
                    throw new \Exception($e->getMessage());
                }
                break;
            case 'Custom_VKP80II':
                sleep(2);
                try {
                    $this->connector = new PrinterFile(array(
                        'directory' => $device->directory, 
                        'MDL' => $device->MDL, 
                        'debug' => $verbose));
                    $printer = new PrinterDevice(array(
                        'connector' => $this->connector, 
                        'debug' => $verbose));
                    $driver = new DriverVKP80II(array(
                        'device' => $printer));
                } catch (\Exception $e) {
                    if (!empty($this->connector)) {
                        $this->connector->close();
                    }
                    throw new \Exception($e->getMessage());
                }
                break;
            case 'Custom_KUBEII':
                sleep(2);
                try {
                    $this->connector = new PrinterFile(array(
                        'directory' => $device->directory, 
                        'MDL' => $device->MDL, 
                        'debug' => $verbose));
                    $printer = new PrinterDevice(array(
                        'connector' => $this->connector, 
                        'debug' => $verbose));
                    $driver = new DriverKUBEII(array(
                        'device' => $printer));
                } catch (\Exception $e) {
                    if (!empty($this->connector)) {
                        $this->connector->close();
                    }
                    throw new \Exception($e->getMessage());
                }
                break;                
        }
        return $driver;
    }

    public function dismissDriver() {
        $this->connector->close();
    }

    public function getStatus($devices_code, $language, $verbose) {
        $out['result'] = true;
        $device = Device::active()->where('code',$devices_code)->first();
        switch ($devices_code) {
            case 'POS_Ingenico_IUP250':
                try {
                    $posDevice = $this->getDriver($devices_code, $verbose);
                    $askEFTPosStatus = new StartOperation;
                    $askEFTPosStatus->set('commandCode', StartOperation::GetEFTPOSStatus)
                            ->set('multilanguageFlag', $language);
                    $posDevice->send($askEFTPosStatus);
                    $out[$devices_code]['result'] = true;
                    $eftPosStatus = $posDevice->receive();
                    $out[$devices_code]['message'] = $eftPosStatus->toJSON();
                    $this->dismissDriver();
                } catch (\Exception $e) {
                    $out[$devices_code]['result'] = false;
                    $out[$devices_code]['message'] = $e->getMessage();
                }
                break;
            case 'POS_Ingenico_SELF5000':
                try {
                    $posDevice = $this->getDriver($devices_code, $verbose);
                    $askEFTPosStatus = new StartOperation;
                    $askEFTPosStatus->set('commandCode', StartOperation::GetEFTPOSStatus)
                            ->set('multilanguageFlag', $language);
                    $posDevice->send($askEFTPosStatus);
                    $out[$devices_code]['result'] = true;
                    $eftPosStatus = $posDevice->receive();
                    $out[$devices_code]['message'] = $eftPosStatus->toJSON();
                    $this->dismissDriver();
                } catch (\Exception $e) {
                    $out[$devices_code]['result'] = false;
                    $out[$devices_code]['message'] = $e->getMessage();
                }
                break;
            case 'POS_Ingenico_SELF5000_S':
                try {
                    $posDevice = $this->getDriver($devices_code, $verbose);
                    $askEFTPosStatus = new StartOperation;
                    $askEFTPosStatus->set('commandCode', StartOperation::GetEFTPOSStatus)
                            ->set('multilanguageFlag', $language);
                    $posDevice->send($askEFTPosStatus);
                    $out[$devices_code]['result'] = true;
                    $eftPosStatus = $posDevice->receive();
                    $out[$devices_code]['message'] = $eftPosStatus->toJSON();
                    $this->dismissDriver();
                } catch (\Exception $e) {
                    $out[$devices_code]['result'] = false;
                    $out[$devices_code]['message'] = $e->getMessage();
                }
                break;
            case 'Custom_KPM150H':
                try {
                    $printer = $this->getDriver($devices_code, $verbose);
                    $printer->getPrinterFullStatus();
                    if ($printer->canPrint()) {
                        $out[$devices_code]['result'] = true;
                    } else {
                        $out[$devices_code]['result'] = false;
                        $out[$devices_code]['reason'] = 'Printer unable to print';
                    }
                    $out[$devices_code]['message'] = $printer->toJSON();
                    $this->dismissDriver();
                } catch (\Exception $e) {
                    $out[$devices_code]['result'] = false;
                    $out[$devices_code]['reason'] = $e->getMessage();
                }
                break;
            case 'Custom_VKP80II':
                try {
                    $printer = $this->getDriver($devices_code, $verbose);
                    $printer->getPrinterFullStatus();
                    if ($printer->canPrint()) {
                        $out[$devices_code]['result'] = true;
                    } else {
                        $out[$devices_code]['result'] = false;
                        $out[$devices_code]['reason'] = 'Printer unable to print';
                    }
                    $out[$devices_code]['message'] = $printer->toJSON();
                    $this->dismissDriver();
                } catch (\Exception $e) {
                    $out[$devices_code]['result'] = false;
                    $out[$devices_code]['reason'] = $e->getMessage();
                }
                break;
            case 'Custom_KUBEII':
                try {
                    $printer = $this->getDriver($devices_code, $verbose);
                    $printer->getPrinterFullStatus();
                    if ($printer->canPrint()) {
                        $out[$devices_code]['result'] = true;
                    } else {
                        $out[$devices_code]['result'] = false;
                        $out[$devices_code]['reason'] = 'Printer unable to print';
                    }
                    $out[$devices_code]['message'] = $printer->toJSON();
                    $this->dismissDriver();
                } catch (\Exception $e) {
                    $out[$devices_code]['result'] = false;
                    $out[$devices_code]['reason'] = $e->getMessage();
                }
                break;
                
        }
        if ($verbose) {
            $m = json_encode($out);
            echo $m;
            echo str_repeat(' ', $this->bufferSize - strlen($m));
        }
        return $out;
    }

    public function pay($devices_code, $amount, $language = StartOperation::Italian, $logs_id = null) {
        
        ob_implicit_flush(true);
        while (ob_get_level()) {
            ob_end_flush();
        }
        $send = true;
        $output = array();
        $out = array();
        try {
            $posDevice = $this->getDriver($devices_code);
        } catch (\Exception $e) {
            $out[$devices_code]['result'] = false;
            $out[$devices_code]['message'] = $e->getMessage();
            return $out;
        }
        $startedTime = false;
        while (true) {
            if ($send) {
                try {
                    $paymentRequest = new StartOperation;
                    $paymentRequest->set('multilanguageFlag', $language)
                            ->set('commandCode', StartOperation::Payment)
                            ->set('activateAsynchronousMessages', Message::YES)
                            ->set('finalAmountToDebit', $amount);
                    $posDevice->send($paymentRequest);
                } catch (\Exception $e) {
                    if (empty($out['receive'])) {
                        $out['result'] = false;
                        $out['message'] = $e->getMessage();
                    }
                    $json = json_encode($out);
                    echo $json;
                    echo str_repeat(' ', $this->bufferSize - strlen($json));
                    $this->dismissDriver();
                    return $json;
                }
                $send = false;
            }
            try {
                $receive = $posDevice->receive();
                $message['type'] = $receive->getType();
                $message['msg'] = $receive->toJSON();
                $out = array();
                $out['result'] = true;
                $out['message'] = $message;
                $json = json_encode($out);
                if ($receive->getType() != Message::FinancialTransactionError || ($receive->getType() == Message::FinancialTransactionError && $receive->get('errorCode') != '90')) {
                    echo $json;
                    echo str_repeat(' ', $this->bufferSize - strlen($json));
                    $output[] = $out;
                    \TicketCenter\Models\Logs::modify($logs_id, $output);
                }
                switch ($receive->getType()) {
                    case Message::AmountRequest:
                              //                  goto esc; // da usare per provare il pos hang up
                        $sendingAmount = new \App\Lib\Ingenico\iSelf\Messages\Commands\SendingAmount; 
                        $sendingAmount->set('amountValue', $amount);
                        $posDevice->send($sendingAmount);
                        break;
                    case Message::AsyncMessage:
                        switch ($receive->get('typeMessageCode')) {
                            case AsyncMessage::StopKeyPressed:
                                $this->abort = true;
                                break;
                            case AsyncMessage::TimeoutExpire:
                                $this->abort = true; 
                                break;
                            case AsyncMessage::EndOfTransaction:
                              //  goto esc; // da usare per provare il pos hang up
                                $startedTime = time();
                                break;
                        }
                        break;
                    case Message::FinancialTransactionEnd:
                     // goto esc; // da usare per provare il pos hang up
                        try {
                            $getEMVTransactionData = new StartOperation;
                            $getEMVTransactionData->set('commandCode', StartOperation::GetEMVTransactionData);
                            $posDevice->send($getEMVTransactionData);
                        } catch (\Exception $e) {
                            if (empty($out['receive'])) {
                                $out['result'] = false;
                                $out['message'] = $e->getMessage();
                            }
                            $json = json_encode($out);
                            echo $json;
                            echo str_repeat(' ', $this->bufferSize - strlen($json));
                            $output[] = $json;
                            $this->dismissDriver();
                            return $output;
                        }
                        break;
                    case Message::ServiceOperation3:
                        $send = true;
                        break;
                    case Message::FinancialTransactionError:
                        switch ($receive->get('errorCode')) {
                            case '90':
                                $askActivateEFTPOS = new StartOperation;
                                $askActivateEFTPOS->set('commandCode', StartOperation::ActivateEFTPOS);
                                $posDevice->send($askActivateEFTPOS);
                                $activateEFTPOS = $posDevice->receive();
                                $send = true;
                                break;
                            default:
                                $this->dismissDriver();
                                return $output;
                        }
                        break;
                    case Message::GetEMVTransactionData:
                        $this->dismissDriver();
                        return $output;
                    case Message::CancelTransaction:
                        // TODO
                        // $this->abort = true;
                        // $this->dismissDriver();
                        // return $output;
                        break;
                        // default:
                        // var_dump($output);
                        // $this->dismissDriver();
                        // return $output;
                }
            } catch (\Exception $e) {
                // empty messages are simply ignored, but not after the POS response timeout!
                if ($startedTime !== false && time() - $startedTime >= $this->posResponseTimeout) {
                    esc:
                    $hangUp = new App\Lib\POS\Ingenico\iSelf\Messages\Responses\HangUp;
                    $message['type'] = $hangUp->getType();
                    $message['msg'] = $hangUp->toJSON();
                    $out = array();
                    $out['result'] = true;
                    $out['message'] = $message;
                    $json = json_encode($out);
                    echo $json;
                    echo str_repeat(' ', $this->bufferSize - strlen($json));
                    $output[] = $out;
                    $this->dismissDriver();
                    return $output;
                }
            }
        }
    }

    public static function formatAmount($amount) {
        $amount = str_replace(",", ".", $amount);
        if (!is_numeric($amount)) {
            return false;
        }
        if (strlen($amount) == 8) {
            return $amount;
        }
        $amount = floor($amount * 100);
        $amount = str_pad($amount, 8, "0", STR_PAD_LEFT);
        return $amount;
    }
}