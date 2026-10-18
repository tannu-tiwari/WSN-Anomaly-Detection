#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>

using namespace ns3;

int main(int argc, char *argv[])
{
    // Enable logging for our simulation
    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    // ----------------------------------------------------
    // 1. CREATE SENSOR NODES
    // ----------------------------------------------------

    NodeContainer nodes;
    nodes.Create(3);

    // ----------------------------------------------------
    // 2. CREATE AD-HOC WIFI NETWORK
    // ----------------------------------------------------

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    YansWifiChannelHelper wifiChannel =
        YansWifiChannelHelper::Default();

    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());

    NetDeviceContainer devices =
        wifi.Install(wifiPhy, wifiMac, nodes);

    // ----------------------------------------------------
    // 3. SET NODE POSITIONS
    // ----------------------------------------------------

    MobilityHelper mobility;

    Ptr<ListPositionAllocator> positionAlloc =
        CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(20.0, 0.0, 0.0));
    positionAlloc->Add(Vector(40.0, 0.0, 0.0));

    mobility.SetPositionAllocator(positionAlloc);

    mobility.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");

    mobility.Install(nodes);

    // ----------------------------------------------------
    // 4. INSTALL INTERNET PROTOCOL STACK
    // ----------------------------------------------------

    InternetStackHelper internet;
    internet.Install(nodes);

    // ----------------------------------------------------
    // 5. ASSIGN IP ADDRESSES
    // ----------------------------------------------------

    Ipv4AddressHelper address;

    address.SetBase(
        "10.1.1.0",
        "255.255.255.0");

    Ipv4InterfaceContainer interfaces =
        address.Assign(devices);

    // ----------------------------------------------------
    // 6. CREATE UDP SERVER
    // ----------------------------------------------------

    UdpEchoServerHelper server(9);

    ApplicationContainer serverApp =
        server.Install(nodes.Get(2));

    serverApp.Start(Seconds(1.0));
    serverApp.Stop(Seconds(10.0));

    // ----------------------------------------------------
    // 7. CREATE UDP CLIENT
    // ----------------------------------------------------

    UdpEchoClientHelper client(
        interfaces.GetAddress(2),
        9);

    client.SetAttribute(
        "MaxPackets",
        UintegerValue(20));

    client.SetAttribute(
        "Interval",
        TimeValue(Seconds(0.5)));

    client.SetAttribute(
        "PacketSize",
        UintegerValue(512));

    ApplicationContainer clientApp =
        client.Install(nodes.Get(0));

    clientApp.Start(Seconds(2.0));
    clientApp.Stop(Seconds(10.0));
     // ----------------------------------------------------
       // ----------------------------------------------------
    // 8. FLOW MONITOR
    // ----------------------------------------------------

    FlowMonitorHelper flowMonitorHelper;

    Ptr<FlowMonitor> flowMonitor =
        flowMonitorHelper.InstallAll();

    // ----------------------------------------------------
    // 9. RUN SIMULATION
    // ----------------------------------------------------

    Simulator::Stop(Seconds(10.0));

    Simulator::Run();

    // ----------------------------------------------------
    // 10. COLLECT FLOW STATISTICS
    // ----------------------------------------------------

    flowMonitor->CheckForLostPackets();

    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(
            flowMonitorHelper.GetClassifier());

    std::map<FlowId, FlowMonitor::FlowStats> stats =
        flowMonitor->GetFlowStats();
    std::ofstream csvFile("data/wsn_metrics.csv");

    csvFile << "flow_id,source,destination,packets_sent,"
           "packets_received,packets_lost,pdr,"
           "throughput_bps,average_delay_seconds\n";
    for (const auto &flow : stats)
    {
        Ipv4FlowClassifier::FiveTuple t =
            classifier->FindFlow(flow.first);

        std::cout << "Flow ID: "
                  << flow.first
                  << std::endl;

        std::cout << "Source: "
                  << t.sourceAddress
                  << std::endl;

        std::cout << "Destination: "
                  << t.destinationAddress
                  << std::endl;

        std::cout << "Packets Sent: "
                  << flow.second.txPackets
                  << std::endl;

        std::cout << "Packets Received: "
                  << flow.second.rxPackets
                  << std::endl;

        std::cout << "Packets Lost: "
                  << flow.second.lostPackets
                  << std::endl;

        double pdr = 0.0;

        if (flow.second.txPackets > 0)
        {
            pdr =
                static_cast<double>(flow.second.rxPackets) /
                flow.second.txPackets * 100.0;
        }

        std::cout << "PDR: "
                  << pdr
                  << "%"
                  << std::endl;

        double throughput = 0.0;

        double simulationTime =
            flow.second.timeLastRxPacket.GetSeconds() -
            flow.second.timeFirstTxPacket.GetSeconds();

        if (simulationTime > 0)
        {
            throughput =
                flow.second.rxBytes * 8.0 /
                simulationTime;
        }

        std::cout << "Throughput: "
                  << throughput
                  << " bit/s"
                  << std::endl;

        double averageDelay = 0.0;

        if (flow.second.rxPackets > 0)
        {
            averageDelay =
                flow.second.delaySum.GetSeconds() /
                flow.second.rxPackets;
        }

        std::cout << "Average Delay: "
                  << averageDelay
                  << " seconds"
                  << std::endl;

        csvFile << flow.first << ","
                << t.sourceAddress << ","
                << t.destinationAddress << ","
                << flow.second.txPackets << ","
                << flow.second.rxPackets << ","
                << flow.second.lostPackets << ","
                << pdr << ","
                << throughput << ","
               << averageDelay
               << "\n";

        std::cout << "-----------------------------"
                  << std::endl;
    }
    csvFile.close();
    // ----------------------------------------------------
    // 11. DESTROY SIMULATION
    // ----------------------------------------------------

    Simulator::Destroy();
    return 0;
}