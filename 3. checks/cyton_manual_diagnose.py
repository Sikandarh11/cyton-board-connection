import argparse
import sys
import time


def safe_text(value):
    return value if value is not None else ""


def list_serial_ports():
    try:
        import serial.tools.list_ports
    except Exception as exc:
        print("[ERROR] pyserial is not available:", exc)
        print("Install with: python -m pip install pyserial")
        return []

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("[INFO] No serial ports found.")
        return []

    print("\n=== Serial Ports Detected ===")
    for idx, p in enumerate(ports, start=1):
        print(f"{idx}. {p.device}")
        print(f"   Description : {safe_text(getattr(p, 'description', ''))}")
        print(f"   Manufacturer: {safe_text(getattr(p, 'manufacturer', ''))}")
        print(f"   HWID        : {safe_text(getattr(p, 'hwid', ''))}")
        print(f"   VID:PID     : {safe_text(getattr(p, 'vid', ''))}:{safe_text(getattr(p, 'pid', ''))}")

    return ports


def score_port(port_info):
    text = " ".join(
        [
            safe_text(getattr(port_info, "device", "")),
            safe_text(getattr(port_info, "description", "")),
            safe_text(getattr(port_info, "manufacturer", "")),
            safe_text(getattr(port_info, "hwid", "")),
        ]
    ).lower()

    keywords = ["openbci", "ftdi", "usb serial", "cp210", "silabs", "ch340"]
    return sum(1 for k in keywords if k in text)


def serial_probe(port, baud=115200, timeout=1.0):
    try:
        import serial
    except Exception as exc:
        return {
            "ok": False,
            "port": port,
            "error": f"pyserial import failed: {exc}",
            "welcome": "",
            "stream_packets": 0,
        }

    result = {
        "ok": False,
        "port": port,
        "error": "",
        "welcome": "",
        "stream_packets": 0,
    }

    try:
        with serial.Serial(port=port, baudrate=baud, timeout=timeout) as ser:
            time.sleep(0.3)
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            # Ask for board welcome/version text.
            ser.write(b"v")
            ser.flush()

            deadline = time.time() + 4.0
            chunks = []
            while time.time() < deadline:
                waiting = ser.in_waiting
                if waiting > 0:
                    chunks.append(ser.read(waiting))
                time.sleep(0.05)

            welcome = b"".join(chunks).decode(errors="ignore")
            result["welcome"] = welcome.strip()

            known_tokens = ["OpenBCI", "ADS1299", "Firmware", "Ganglion", "Cyton"]
            welcome_ok = any(token in welcome for token in known_tokens)

            # Try a short stream check.
            ser.write(b"b")
            ser.flush()
            time.sleep(1.0)
            raw = ser.read(ser.in_waiting or 512)
            ser.write(b"s")
            ser.flush()

            packets = raw.count(b"\xA0")
            result["stream_packets"] = packets
            result["ok"] = welcome_ok or packets > 0

            if not result["ok"]:
                result["error"] = "No welcome text and no streaming packet header detected"

    except Exception as exc:
        result["error"] = f"Serial open/probe failed: {exc}"

    return result


def brainflow_probe(port, board_id=0):
    info = {"ok": False, "error": ""}

    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams
    except Exception as exc:
        info["error"] = f"BrainFlow import failed: {exc}"
        return info

    params = BrainFlowInputParams()
    params.serial_port = port

    board = None
    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        info["ok"] = True
    except Exception as exc:
        info["error"] = str(exc)
    finally:
        if board is not None:
            try:
                if board.is_prepared():
                    board.release_session()
            except Exception:
                pass

    return info


def print_issue_hints(serial_result, brainflow_result=None):
    print("\n--- Diagnosis ---")

    if serial_result["ok"]:
        print("[PASS] Serial-level response detected from board/dongle path.")
    else:
        print("[FAIL] Serial-level Cyton response not confirmed.")
        print("Likely causes:")
        print("1) Board is OFF or battery is too low")
        print("2) Dongle and Cyton are not paired / wrong radio channel")
        print("3) Wrong COM port selected")
        print("4) Another app is using the COM port")

    if brainflow_result is not None:
        if brainflow_result["ok"]:
            print("[PASS] BrainFlow prepare_session succeeded.")
        else:
            print("[FAIL] BrainFlow prepare_session failed.")
            print("BrainFlow error:")
            print(brainflow_result["error"])
            print("Common fixes:")
            print("1) Confirm board type is Cyton (board_id=0)")
            print("2) Close OpenBCI GUI / serial monitor before running script")
            print("3) Replug dongle and power-cycle Cyton")


def choose_ports(all_ports, requested_port=None):
    if requested_port:
        return [requested_port]

    if not all_ports:
        return []

    scored = sorted(all_ports, key=score_port, reverse=True)
    return [p.device for p in scored]


def main():
    parser = argparse.ArgumentParser(description="Manual Cyton wireless diagnostics")
    parser.add_argument("--port", default="", help="Specific COM port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--board-id", type=int, default=0, help="BrainFlow board_id (Cyton=0)")
    parser.add_argument("--list-only", action="store_true", help="Only list serial ports")
    parser.add_argument("--no-brainflow", action="store_true", help="Skip BrainFlow probe")
    args = parser.parse_args()

    ports_info = list_serial_ports()

    if args.list_only:
        return 0

    probe_ports = choose_ports(ports_info, requested_port=args.port.strip() or None)

    if not probe_ports:
        print("\n[STOP] No ports to test.")
        return 1

    print("\n=== Probe Order ===")
    for i, p in enumerate(probe_ports, start=1):
        print(f"{i}. {p}")

    any_ok = False

    for p in probe_ports:
        print("\n" + "=" * 70)
        print(f"Testing port: {p}")
        sres = serial_probe(p, baud=args.baud)

        print(f"Serial result : {'PASS' if sres['ok'] else 'FAIL'}")
        if sres["welcome"]:
            print("Welcome text  :")
            print(sres["welcome"][:500])
        print(f"Stream packets: {sres['stream_packets']}")
        if sres["error"]:
            print(f"Serial error  : {sres['error']}")

        bres = None
        if not args.no_brainflow:
            bres = brainflow_probe(p, board_id=args.board_id)
            print(f"BrainFlow     : {'PASS' if bres['ok'] else 'FAIL'}")
            if bres["error"]:
                print(f"BrainFlow err : {bres['error']}")

        print_issue_hints(sres, bres)

        if sres["ok"] and (bres is None or bres["ok"]):
            any_ok = True
            print("\n[READY] This port looks good for Cyton streaming.")
            break

    print("\n" + "=" * 70)
    if any_ok:
        print("Final result: At least one working Cyton path found.")
        return 0

    print("Final result: Could not confirm Cyton wireless link on tested ports.")
    print("Next actions:")
    print("1) Re-pair dongle and Cyton radio channel")
    print("2) Fully charge/replace Cyton battery")
    print("3) Re-test with OpenBCI GUI to isolate hardware vs script")
    return 2


if __name__ == "__main__":
    sys.exit(main())
