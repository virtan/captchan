#include <iostream>

class progress {
    public:
        void start() {
            std::cout << "[";
        }

        void update(float progress) {
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();
        }

        void done() {
            std::cout << std::endl;
        }

    private:
        const int barWidth = 70;
};
