#include "ascii.h"
#include <sstream>
#include <iomanip>

namespace bg {

    static void print_points_row(std::ostringstream& os,
        const std::array<uint8_t, 24>& a,
        int l, int r) {
        for (int i = l; i <= r; ++i) {
            os << std::setw(2) << int(a[i]) << ' ';
        }
    }

    std::string to_ascii(const State& s, const Dice& d) {
        std::ostringstream os;
        os << "Dice: " << int(d.a) << "," << int(d.b) << " | ply=" << int(s.ply) << "\n";
        os << "Mine  bar=" << int(s.bar) << " off=" << int(s.off) << "\n";
        os << "Opp   bar=" << int(s.opp_bar) << " off=" << int(s.opp_off) << "\n";

        os << "Mine points [23..12]: ";
        for (int i = 23; i >= 12; --i) os << std::setw(2) << int(s.points[i]) << ' ';
        os << "\n";
        os << "Mine points [11..0 ]: ";
        for (int i = 11; i >= 0; --i) os << std::setw(2) << int(s.points[i]) << ' ';
        os << "\n";

        os << "Opp  points [23..12]: ";
        for (int i = 23; i >= 12; --i) os << std::setw(2) << int(s.opp_points[i]) << ' ';
        os << "\n";
        os << "Opp  points [11..0 ]: ";
        for (int i = 11; i >= 0; --i) os << std::setw(2) << int(s.opp_points[i]) << ' ';
        os << "\n";

        return os.str();
    }

} // namespace bg