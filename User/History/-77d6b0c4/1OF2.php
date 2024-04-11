<?php

namespace App\Http\Controllers\TVM\POS\Ingenico;

use App\Http\Controllers\TVM\POS\Ingenico\Message;
# use App\Http\Controllers\TVM\POS\Ingenico\iSelf\Messages\Ack;

class Device {

    protected $connector;
    protected $terminalIdentifier;
    protected $debug = true;
    protected $ioTimeout = 5000;
    protected $bufferSize = 4096;

    public function __construct($options = array()) {
        if (!empty($options)) {
            foreach ($options as $option => $value) {
                switch ($option) {
                    case 'connector':
                        $this->setConnector($value);
                        break;
                    case 'terminalIdentifier':
                        $this->setTerminalIdentifier($value);
                        break;
                    case 'debug':
                        $this->setDebug($value);
                        break;
                    case 'ioTimeout':
                        $this->setIOTimeout($value);
                        break;
                }
            }
            try {
                $this->debug("Attempting to open the connection....\n");
                $this->connector->open();
                $this->debug("Open succeeded.\n");
            } catch (\Exception $e) {
                $this->debug($e->getMessage());
                $this->debug("Open failed.\n");
                throw new \Exception($e->getMessage());
            }
        }
        $this->bufferSize = ini_get('output_buffering');
        if ($this->debug) {
            ob_implicit_flush(true);
            while (ob_get_level()) {
                ob_end_flush();
            }
        }
    }

    public function __destruct() {
        if (!empty($this->connector)) {
            $this->close();
        }
    }

    public function setConnector($connector) {
        $this->connector = $connector;
        return $this;
    }

    public function setTerminalIdentifier($terminalIdentifier) {
        $this->terminalIdentifier = $terminalIdentifier;
        return $this;
    }

    public function setDebug($debug) {
        $this->debug = $debug;
        return $this;
    }

    public function setIOTimeout($ioTimeout) {
        $this->ioTimeout = $ioTimeout;
    }

    public function send(Message $message, $debug = true) {
        $message->set('terminalIdentifier', $this->terminalIdentifier);
        if ($debug) {
            switch ($message->getType()) {
                case ACK:
                    $this->debug("Attempting to send an ACK message:\n");
                    break;
                case NAK:
                    $this->debug("Attempting to send a NAK message':\n");
                    break;
                default:
                    $this->debug("Attempting to send a message:\n");
            }
            $this->debug($message->debug());
        }
        $attempt = 1;
        $sent = false;
        while (!$sent && $attempt <= 3) {
            try {
                if ($debug) {
                    $this->debug("Attempt $attempt\n");
                }


                // echo $message->getStringToSend() . '<br/>';


                $this->connector->write($message->getStringToSend());
            } catch (\Exception $e) {
                if ($debug) {
                    $this->debug($e->getMessage());
                    $this->debug("Send failed.\n");
                }
                throw new \Exception($e->getMessage());
            }
            if ($message->getType() == ACK) {
                $sent = true;
            } else {
                try {
                    $response = $this->receive(1024, false);
                    if ($response->isAck()) {
                        $sent = true;
                    } else {
                        $attempt++;
                        if ($debug) {
                            $this->debug("Attempt failed: no ack back ({$response->getType()} instead)\n");
                        }
                    }
                } catch (\Exception $e) {
                    $attempt++;
                    if ($debug) {
                        $this->debug("Attempt failed: {$e->getMessage()}\n");
                    }
                }
            }
        }
        if (!$sent && $attempt > 3) {
            $debugMessage = "Three attempts to send a message have failed\n";
            if ($debug) {
                $this->debug($debugMessage);
            }
            throw new \Exception($debugMessage);
        }

        if ($message->getType() != ACK && $response->isAck()) {
            if ($debug) {
                $this->debug("Send succeeded.\n");
            }
        }
        return $this;
    }

    public function simpleSend(Message $message) {
        $this->connector->write($message->getStringToSend());
        return $this;
    }

    public function receive($length = 0, $debug = true) {
        if ($debug) {
            $this->debug("Attempting to receive a message:\n");
        }
        $start_time = time();
        $results = '';
        $waiting = true;
        while ($waiting) {
            try {
                $result = $this->connector->read($length);
            } catch (Exception $e) {
                if ($debug) {
                    $this->debug($e->getMessage());
                }
                throw new \Exception($e->getMessage());
            }
            $results.=$result;    
            $message = Message::decipher($results);
            if ($message !== false) {
                $waiting = false;
            }
            if ($waiting) {
                $waiting = (time() - $start_time) < $this->ioTimeout / 1000;
            }
        }
        if ($message === false) {
            if (strlen($results) == 0) {
                $debugMessage = "Message empty.\n";
            } else {
                $debugMessage = "Message deciphering failed: $results\n";
            }
            if ($debug) {
                $this->debug($debugMessage);
            }
            throw new \Exception($debugMessage);
        } else {
            if ($debug) {
                $this->debug("\t" . $message->debug());
            }
            if (!$message->isAck() && !$message->isAsync()) {
                try {
                    $this->send(new \Messages\Ack, false);
                } catch (\Exception $e) {
                    throw new \Exception($e->getMessage());
                }
            }
        }
        return $message;
    }

    public function simpleReceive($length = 0) {
        $start_time = time();
        $result = '';
        $waiting = true;
        while ($waiting) {
            try {
                $result = $this->connector->read($length);
            } catch (Exception $e) {
                throw new \Exception($e->getMessage());
            }
            if (strlen($result) > 0) {
                $message = Message::decipher($result);
                return $message;
            }
            $waiting = (time() - $start_time) < $this->ioTimeout / 1000;
        }
        return false;
    }

    public function flush($timeout = 10) {
        $this->debug('Start Flushing');
        $nakReceived = false;
        while (!$nakReceived) {
            try {
                $erroneous = new \POS\Ingenico\iSelf\Messages\Commands\Erroneous;
                $this->debug('Send Erroneous message: NAK expected');
                $this->simpleSend($erroneous);
            } catch (\Exception $e) {
                
            }
            $start_time = time();
            $stop = false;
            while (!$stop) {
                try {
                    if ((time() - $start_time) > $timeout) {
                        $stop = true;
                        $this->debug('Timeout Occurred');
                    }
                    $m = $this->receive();
                    if ($m !== false && $m->getType() == Message::NAK) {
                        $stop = true;
                        $nakReceived = true;
                    }
                } catch (\Exception $e) {
                    // empty messages are simply ignored
                }
            }
        }
        $this->debug('End Flushing');
    }

    public function close() {
        try {
            $this->debug("Attempting to close the connection....\n");
            $this->connector->close();
            $this->debug("Close succeeded.\n");
        } catch (\Exception $e) {
            $this->debug($e->getMessage());
            $this->debug("Close failed.\n");
            throw new \Exception($e->getMessage());
        }
    }

    protected function debug($message) {
        if ($this->debug) {
            $m = json_encode(array('m' => $message));
            echo $m;
            echo str_repeat(' ', $this->bufferSize - strlen($m));
        }
    }

}
