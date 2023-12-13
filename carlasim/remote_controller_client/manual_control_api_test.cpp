#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <string>
#include <mosquittopp.h>
#include <signal.h>
#include <memory>
#include <vehicle_hal_carla.h>
#include "file_utils.h"
#include <nlohmann/json.hpp>
using nlohmann::json;

#define BROKER_IP "127.0.0.1"
#define BROKER_PORT 1883

std::unique_ptr<VehicleHALCarlaImpl> hal;

char menu()
{
    std::string hs;
    std::string ms;
    std::string ack;

    if (hal != nullptr && hal->getLastSteeringAngle() < 0)
        hs = "[-]";
    else
        hs = "[+]";

    if (hal != nullptr && hal->getLastEnginePower() < 0)
        ms = "[-]";
    else
        ms = "[+]";

    // system("clear");
    printf("Hardware testing menu\n\n");
    printf("                   (w)       forward pwr++\n");
    printf("left pwr++   (a)   (q)   (d)      right pwr++\n");
    printf("                   (s)       backward pwr++\n");
    printf("\n\n");

    if (hal != nullptr)
    {
        printf(">>>> moving wheeldrive: H: %s, power: %d\n", ms.c_str(), hal->getLastEnginePower());
        printf("<--> heading: H: %s, angle: %d\n", hs.c_str(), hal->getLastSteeringAngle());
    }
    else
    {
        printf(">>>> moving wheeldrive: H: ?, power: ?\n");
        printf(">>>> heading: H: ?, power: ?\n");
    }

    printf("\n\n");
    printf("(r) reset\n");
    printf("(q) stop\n");
    printf("(p) AUTONOMOUS MODE ON\n");
    printf("(Esc) to quit\n\n");
    return getchar();
}

unsigned int setupTerminal()
{
    static struct termios term_flags;
    tcgetattr(STDIN_FILENO, &term_flags);
    unsigned int oldFlags = term_flags.c_lflag;
    // newt = oldt;
    term_flags.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &term_flags);
    return oldFlags;
}

void restoreTerminal(int oldFlags)
{
    static struct termios term_flags;
    tcgetattr(STDIN_FILENO, &term_flags);
    term_flags.c_lflag = oldFlags;
    tcsetattr(STDIN_FILENO, TCSANOW, &term_flags);
}

class RemoteController : public mosqpp::mosquittopp
{
private:
    char *host_addr;
    int port;
    std::thread *mqttLoopThr;

    void runMqttLoop()
    {
        loop_forever();
    }

public:
    RemoteController()
    {
        // this->username_pw_set("Vehicle", "435FKDVpp48ddf");
        this->mqttLoopThr = new std::thread(&RemoteController::runMqttLoop, this);
        connect(BROKER_IP, BROKER_PORT, 60);
    }

    ~RemoteController()
    {
        disconnect();
        free(host_addr);
    }

    void requestForwardIncrement()
    {
        hal->setEnginePower(hal->getLastEnginePower() + 25);
    }

    void requestForwardDecrement()
    {
        hal->setEnginePower(hal->getLastEnginePower() - 25);
    }

    void requestLeftIncrement()
    {
        hal->setSteeringAngle(hal->getLastSteeringAngle() - 5);
    }

    void requestRightIncrement()
    {
        hal->setSteeringAngle(hal->getLastSteeringAngle() + 5);
    }

    void requestStop()
    {
        hal->setSteeringAngle(0);
        hal->setEngineStop();
    }

    void requestReset()
    {
        hal->resetController();
    }

    void resumeAutonomousDriving()
    {
        json j{
            {"action", "resume_autonomous_driving"},
            {"value", true}};

        int id = 1000;

        std::string payload = j.dump();
        publish(&id, "/virtual_car/action", payload.size(), payload.c_str(), 1);
    }

    void on_message(const struct mosquitto_message *message)
    {
    }
};

void cancelHandler(int s)
{
}

int main(int argc, char *argv[])
{
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = cancelHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    hal = std::make_unique<VehicleHALCarlaImpl>(BROKER_IP, BROKER_PORT);

    // client for the MQTT messaging bus

    RemoteController controller;
    bool run = true;

    auto flags = setupTerminal();

    while (run)
    {
        switch (menu())
        {
        case 'w':
            controller.requestForwardIncrement();
            break;
        case 's':
            controller.requestForwardDecrement();
            break;
        case 'a':
            controller.requestLeftIncrement();
            break;
        case 'd':
            controller.requestRightIncrement();
            break;
        case 'q':
            controller.requestStop();
            break;
        case 'r':
            controller.requestReset();
            break;
        case 'p':
            controller.resumeAutonomousDriving();
            break;
        case 27:
            run = false;
            break;
        default:
            break;
        }

        if (!run)
            break;
    }

    restoreTerminal(flags);
    return 0;
}