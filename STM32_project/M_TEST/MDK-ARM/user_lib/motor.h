#ifndef __MOTOR_H
#define __MOTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"  
#include <stdint.h>
#include <stdbool.h>



void Motor_Init(void);
void Motor_Speed_Percent(uint8_t device, uint8_t percent, int8_t direction);

#ifdef __cplusplus
}
#endif

#endif /* __MOTOR_H */
